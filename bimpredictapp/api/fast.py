import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from os.path import dirname, abspath, join

dirname = dirname(dirname(abspath(__file__)))

#import the prediction module
#from bimpredictapp.interface.main import pred

app = FastAPI()

# Allowing all middleware is optional, but good practice for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

#Example : http://127.0.0.1:8000/bimpred?file=someurl
@app.get("/bimpred")
def bimpred(
        file: str,  # the url of the excel file
    ):
    """
    Call api by file= the url / or path for now ..

    """
    #verify sheets
    #file to test 'RawData - 25NBES1-8010-PRO-BIM-ASPC-MGO-00A-TN-002-0-25nBES1_8010_PRO_STR_ASPC_MAQ_00A_0.xlsx'
    file_name = file
    file_url = join(dirname, 'data/raw/', file_name)

    #sheets to check = ['murs', 'sols', 'poutres', 'poteaux']

    try:
        file = pd.ExcelFile(file_url)
        sheet_names =  file.sheet_names

        for sheet in sheet_names:
            if sheet.lower() in ['murs', 'sols', 'poutres', 'poteaux']:
                results = 'predicted ' #pred(file)
                return str(sheet_names)

    except Exception as e:
        return f"Error loading sheets from file: {e}"

@app.get("/")
def root():
    return {
    'greeting': 'Hello'
    }
