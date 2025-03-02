from fastapi import FastAPI, HTTPException, UploadFile, File, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import pandas
import tensorflow as tf
import pdfplumber
import requests
from fastapi import FastAPI
import nest_asyncio
from pyngrok import ngrok
import uvicorn
import io
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import re
from dotenv import load_dotenv

load_dotenv()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"], 
)


def get_soil_data():
    extracted_values = dict()
    with pdfplumber.open('/content/SoilReport.pdf') as file:
      table = file.pages[0].extract_tables()[0]
      text = file.pages[0].extract_text()  
      # print("Extracted Text:\n", text)
      
      soil_type_match = re.search(r"Sample Description:\s*(.*) soil", text)
      soil_type = soil_type_match.group(1).strip() if soil_type_match else "Not Found"
      if soil_type != 'Not Found':
        soil_type += " soil"
        extracted_values['soil_type'] = soil_type

      print("\nExtracted Soil Type:", soil_type)
      
      
      # print(f"table : {table} tables : {len(table)} tablel[] : {table[0][:]}")
      headers = table[0]
      data = table[1:]
      # print(f"headers : {headers} data : {data}")

      for row in data:
          parameter = row[1].strip() 
          value = float(row[3].strip()) 
          if "pH" in parameter:
              extracted_values["pH"] = value
          elif "Available Nitrogen" in parameter:
              extracted_values["N"] = value
          elif "Available Phosphorus" in parameter:
              extracted_values["P"] = value
          elif "Available Potassium" in parameter:
              extracted_values["K"] = value
    print("\nExtracted Soil Data:")
    print(f"pH: {extracted_values.get('pH', 'Not Found')}")
    print(f"Nitrogen (N): {extracted_values.get('Nitrogen (N)', 'Not Found')} ppm")
    print(f"Phosphorus (P): {extracted_values.get('Phosphorus (P)', 'Not Found')} ppm")
    print(f"Potassium (K): {extracted_values.get('Potassium (K)', 'Not Found')} ppm")
    return extracted_values


def get_weather(location):
    api_url = f"http://api.weatherapi.com/v1/forecast.json?key=6bac00c67d4144e5ad2180607240809&q={location[0]},{location[1]}&days=1&aqi=no&alerts=no"
    print(api_url) 
    response = requests.get(api_url)
    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail="Error fetching weather data")
    weather_data = response.json()
    current_weather = weather_data.get('current', {})
    temperature = current_weather.get('temp_c', None)
    humidity = current_weather.get('humidity', None)
    precipitation = current_weather.get('precip_mm', None)
    data = {
        'Temperature': temperature,
        'Humidity': humidity,
        'Moisture': precipitation
    }
    print(data)
    return data

def preprocess_data(data):
    soil_encoder = LabelEncoder()
    data["Soil Type"] = soil_encoder.fit_transform(data["Soil Type"])
    print(f"soil : {data}")

    crop_encoder = LabelEncoder()
    data["Crop Type"] = crop_encoder.fit_transform(data["Crop Type"])
    print(f"crop : {data}")

    scaler = MinMaxScaler()
    numeric_features = ['Temperature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']
    data[numeric_features] = scaler.fit_transform(data[numeric_features])
    print(f"full : {data}")

    return data


app = FastAPI()

@app.get("/")
def home():
    return {"message": "Hello from FastAPI on Colab!"}

@app.post("/fertilizer-prediction/")
async def predict_fertilizer(
    file : UploadFile = File(...),
    lat: float = Query(..., description="Latitude of the location"), 
    lon: float = Query(..., description="Longitude of the location"),
    crop_type: str = Query(..., description="Type of the crop")
    ):
    print(f"lattitude : {lat} \nlongitude : {lon} \ncroptype : {crop_type}")
    # pdf_content = await file.read()
    # print(f"Read file : {pdf_content}")
    # soil_data = get_soil_data(pdf_content)
    # soil_type = soil_data["soil_type"]
    # n = soil_data["N"]
    # p = soil_data["P"]
    # k = soil_data["K"]

    # weather_data = get_weather([lat, lon])
    # t = weather_data['Temperature']
    # h = weather_data['Humidity']
    # m = weather_data['Moisture']

    # data = {
    #     "Temperature": [t],
    #     "Humidity": [h],
    #     "Moisture": [m],
    #     "Soil Type": [soil_type],
    #     "Crop Type": [crop_type],
    #     "Nitrogen": [n],
    #     "Potassium": [k],
    #     "Phosphorous": [p],
    # }
    # print(f"data : {data}")
    # # df = pandas.DataFrame(data)
    # # print(f"df : {df}") 

    # # processed_data = preprocess_data(df)
    # # print(f"process data : {processed_data}")

    # # pred = model.predict(processed_data)
    # # print(f"pred : {pred}")

    # # max_index = np.argmax(pred[0])

    return JSONResponse(content={"Response": "Function completed"})


nest_asyncio.apply()

# Run the server
public_url = ngrok.connect(8000)
print("FastAPI is running on:", public_url)

uvicorn.run(app, host="0.0.0.0", port=8000)
