from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from joblib import load
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import tensorflow.keras as keras
import numpy as np
import pandas as pd
from pprint import pprint
from pydantic import BaseModel
import os

# Resample to 200x200 based on model
TARGET_SIZE = (200, 200)

# Load in model e.g. model.02-0.0273.h5
model = keras.models.load_model('model.02-0.0273.h5')

# Using defined scaler and mapping
sc = load('parser/std_scaler.bin')
mapping = pd.read_csv('parser/datamapping.csv')
mapping = {col: np.array(mapping[col]) for col in mapping.columns}
pprint(mapping)

app = FastAPI()

origins = [
    "http://crash.severity.ml",
    "https://crash.severity.ml",
    "http://localhost",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/images", StaticFiles(directory="images"), name="images")

@app.get("/list")
def list_images():
    list = os.listdir("images")
    list.remove(".gitignore")
    return list

@app.post("/upload")
def import_file_post(file: UploadFile = File(...)):
    file_location = f"images/{file.filename}"
    with open(file_location, "wb+") as file_object:
        file_object.write(file.file.read())
    return {"filename": file.filename}


class Options(BaseModel):
    filename: str
    weather: str
    brightness: float
    speed: int


@app.post("/predict")
def predict(options: Options):
    image = load_image("images/"+options.filename)
    result = prediction(image, options)
    return {"severity": result.item()}


def load_image(filepath, target_size=TARGET_SIZE):
    return img_to_array(load_img(filepath, target_size=target_size))


def prediction(image, options: Options):
    key = options.weather
    brightness = options.brightness
    speed = options.speed

    # Load, rescale, and reshape image input
    image /= 255

    weather_encoding = mapping[key]
    print(weather_encoding)
    images = np.array([image])
    arr = np.array([brightness, speed])
    inputs = [np.concatenate([weather_encoding, arr])]
    inputs = sc.transform(inputs)
    return model.predict([images, inputs])[0][0]
