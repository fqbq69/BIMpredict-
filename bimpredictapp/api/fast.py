import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


#import verification module
from bimpredictapp.api.verify import get_sheets

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

#Example : http://127.0.0.1:8000/predict?file=someurl&sheets=someparams
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
    sheets = get_sheets(file)

    for sheet in sheets:
        if sheet in ['Murs', 'Sols', 'Poutres', 'Poteaux']:
            results = pred(file)
            return results

        else:
            return 'Sheets Are not OK! Please check your file'

    #predict will return a url to download the predicted excel file

@app.get("/")
def root():
    return {
    'greeting': 'Hello'
    }
