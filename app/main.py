import os
import sys
import logging
import pathlib

import numpy as np
import torch
from transformers import *
from fastapi import FastAPI, File, UploadFile, HTTPException
import onnxruntime as ort
import uvicorn

from utils import (
    set_seed,
    get_prediction,
    Discriminator
)

app = FastAPI()

ROOT_PATH = pathlib.Path(__file__).resolve().parent.parent

model_file_path = os.path.join(ROOT_PATH, "models", "quantizebert1.onnx")
model = ort.InferenceSession(model_file_path)

Discriminator_file_path = os.path.join(ROOT_PATH, "models", "discriminator")
discriminator = torch.load(Discriminator_file_path, map_location = "cpu")
discriminator.eval()

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

@app.post("/predict")
def predict(text: str):    
    try:
        predicted_class = get_prediction(model, tokenizer,
             text, discriminator)
        
        logging.info(f"Predicted Class: {predicted_class}")

        return {            
            "class": predicted_class,
            "status_code": 200
        }

    except Exception as error:
        logging.exception(error)
        e = sys.exc_info()[1]
        raise HTTPException(status_code=500, detail=str(e))