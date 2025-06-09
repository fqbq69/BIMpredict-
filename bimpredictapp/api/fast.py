import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


#import verification module
from bimpredictapp.api.verify import

#imort the prediction module
from bimpredictapp.interface.main import pred


app = FastAPI()

# Allowing all middleware is optional, but good practice for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# http://127.0.0.1:8000/predict?file=someurl&sheets=someparams
@app.get("/predict")
def predict(
        file: str,  # the url of the excel file
        sheets: dict,
    ):
    """
    Load the url and the sheets names into the api to connect with the load function in main module.

    """
    #import excel file: load file to a url file host


    #verify sheets



    #predict will return a url to download the predicted excel file
    results = pred(file, sheets)

    return results


@app.get("/")
def root():
    return {
    'greeting': 'Hello'
    }
