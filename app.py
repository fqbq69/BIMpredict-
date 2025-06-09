import streamlit as st
import streamlit.components.v1 as components

import numpy as np
import pandas as pd
import requests
from datetime import datetime




# upload the Excel file to my cloud URL
address1 = "60 Tiffany Pl, Brooklyn, NY 11231, USA"


label = "Please upload your Excel File:"


def upload_file(url) -> tuple:
    '''
    This function returns the url and the sheets names to the app interface to get them verified
    '''
    try:
        #uploading and run the do thd do
        #do the do
        pass
    except ValueError:
        print("error loading your file..")

#verify sheets:

label = "Please Verify sheet names in your Excel File:"

def verify_sheets(sheet_names:dict) -> dict:
    '''
    This function will place the sheets found in the excel files in an input form,
        for the user to confirm. if a sheet has another name it will be placed as a warning:
        Please rename your sheets as the following: Murs, Sols, Poteaux, Poutres.
    '''
    #we can translate to french if OK

    sheet_murs = st.text_input('Walls Sheet', value=sheet_names{'murs'})
    sheet_sols = st.text_input('Platform Sheet', value=sheet_names{'sols'})
    sheet_poteaux = st.text_input('Posts Sheet', value=sheet_names{'murs'})
    sheet_Poutres = st.text_input('Beams Sheet', value=sheet_names{'poutres'})

    pass


#calling the Taxifare API
def call_api():
    if check_date(pickup_date) & check_time(pickup_time):
        pass
    else:
        return "Bad Date or Time format!"

    pickup_latitude, pickup_longitude = extract_lat_long_via_address(pickup_adress)
    dropoff_latitude, dropoff_longitude = extract_lat_long_via_address(dropoff_adress)

    api_base = 'https://taxifare.lewagon.ai//predict?'

    taxi_api_call = f'{api_base}pickup_datetime={pickup_date}%20\
        {pickup_time}&pickup_longitude=\
        {pickup_latitude}&pickup_latitude=\
        {pickup_longitude}&dropoff_longitude=\
        {dropoff_latitude}&dropoff_latitude=\
        {dropoff_longitude}&passenger_count=\
        {passenger_count}'

    return taxi_api_call

if st.button("Predict"):
    st.write(call_api())
else:
    st.write("Waiting for user input")
