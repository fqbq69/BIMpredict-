import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from os.path import dirname, abspath, join
from fastapi import UploadFile, File
import shutil

dirname = dirname(dirname(abspath(__file__)))

#import the prediction module
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


@app.get("/bimpred")
def bimpred(
        file: str,  #Example : http://127.0.0.1:8000/bimpred?file=file01.xlsx
    ):
    """
    Call api by file= filename
    the file should exist in the data/raw folder

    """
    #verify sheets
    file_url = join(dirname, 'data/raw/', file)

    #sheets to check = ['murs', 'sols', 'poutres', 'poteaux']
    try:
        file = pd.ExcelFile(file_url)
        sheet_names =  file.sheet_names

        for sheet in sheet_names:
            if sheet.lower() in ['murs', 'sols', 'poutres', 'poteaux']:
                results = 'predicted ' #pred(file)
                return {
                        'API respons': 'URL to predicted the excel file'
                }
            else:
                return {
                        'API respons': f'{sheet} is not an expected sheet name!'
                }

    except Exception as e:
        return f"Error loading sheets from file: {e}"

@app.post("/upload_excel")
async def upload_excel(file: UploadFile = File(...)):
    """
    Uploads an Excel file and saves it to /data/raw directory.
    """
    # Usage Example: curl -F "file=@/home/samer/code/test_file.xlsx " http://127.0.0.1:8004/upload_excel

    save_path = join(dirname, 'data/raw', file.filename)
    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return {"filename": file.filename, "status": "saved"}

@app.get("/")
def root():
    return {
    'API': 'please read the full documentation'
    }
