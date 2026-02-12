import requests
import sys

# -----------------------------------------
# API URL
# -----------------------------------------
API_URL = "http://127.0.0.1:8000/predict"

# -----------------------------------------
# Sample Payload (Must match inference.py schema)
# -----------------------------------------
payload = {
    "data": {
        "Type": "L",
        "Air temperature [K]": 298.1,
        "Process temperature [K]": 308.6,
        "Rotational speed [rpm]": 1551,
        "Torque [Nm]": 42.8,
        "Tool wear [min]": 0
    }
}

# -----------------------------------------
# Smoke Test Execution
# -----------------------------------------
try:
    response = requests.post(API_URL, json=payload)

    if response.status_code == 200:
        print(" Smoke Test Passed")
        print("Response:", response.json())
    else:
        print(f" Smoke Test Failed: Status code {response.status_code}")
        print("Response:", response.text)
        sys.exit(1)

except requests.exceptions.ConnectionError:
    print(" Smoke Test Failed: Unable to connect to API.")
    print("Make sure the API server is running.")
    sys.exit(1)

except Exception as e:
    print(" Smoke Test Failed:", str(e))
    sys.exit(1)
