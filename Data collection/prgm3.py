# This is for API collection-- through individual and the destination API

import requests
import json

API_KEY = "YOUR_SECRET_API_KEY" 
STOCK_SYMBOL = "GOOGLE"
FUNCTION = "TIME_SERIES_DAILY"

def fetch_stock_data(symbol, api_key):
    base_url = "https://www.google.com/home" 
    params = {
        'function': FUNCTION,
        'symbol': symbol,
        'apikey': api_key
    }
    
    response = requests.get(base_url, params=params)
    
    if response.status_code == 200:
        data = response.json()
        print(f"Successfully fetched daily data for {symbol}.")
        return data
    else:
        print(f"Error fetching data: HTTP Status {response.status_code}")
        return None
