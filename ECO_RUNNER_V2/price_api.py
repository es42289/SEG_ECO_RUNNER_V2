import requests

url = "https://query1.finance.yahoo.com/v8/finance/chart/CL=F"

params = {
    "region": "US",
    "lang": "en-US",
    "includePrePost": "false",
    "interval": "1d",
    "range": "5d"
}

response = requests.get(url, params=params)
data = response.json()

# Get the most recent closing price
timestamps = data["chart"]["result"][0]["timestamp"]
closes = data["chart"]["result"][0]["indicators"]["quote"][0]["close"]
latest_price = closes[-1]

print(f"Latest CL=F (WTI Crude Oil) Price: ${latest_price:.2f}")
