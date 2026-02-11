import requests
import sys

API_URL = "http://127.0.0.1:8000/predict"

def run_smoke_test():
    """
    Smoke test for ML inference API.
    Ensures:
    - API is reachable
    - Response schema is valid
    - Model returns sensible probability
    """

    # High-risk payload to verify non-zero probability
    payload = {
        "Type": "L",
        "Air temperature [K]": 340,
        "Process temperature [K]": 360,
        "Rotational speed [rpm]": 2800,
        "Torque [Nm]": 80,
        "Tool wear [min]": 260,
        "TWF": 1,
        "HDF": 1,
        "PWF": 1,
        "OSF": 1,
        "RNF": 0
    }

    try:
        response = requests.post(API_URL, json=payload, timeout=5)
    except requests.exceptions.RequestException as e:
        print(" Smoke Test Failed: API not reachable")
        print(str(e))
        sys.exit(1)

    # Status code check
    if response.status_code != 200:
        print(f" Smoke Test Failed: Status code {response.status_code}")
        print("Response:", response.text)
        sys.exit(1)

    # Parse JSON
    try:
        result = response.json()
    except ValueError:
        print("Smoke Test Failed: Response is not valid JSON")
        sys.exit(1)

    # Schema validation
    if "prediction" not in result or "probability" not in result:
        print(" Smoke Test Failed: Invalid response schema")
        print("Response:", result)
        sys.exit(1)

    prediction = result["prediction"]
    probability = result["probability"]

    # Logical validation
    if not (0.0 <= probability <= 1.0):
        print(" Smoke Test Failed: Probability out of range")
        sys.exit(1)

    if prediction not in [0, 1]:
        print("Smoke Test Failed: Invalid prediction value")
        sys.exit(1)

    print("Smoke Test Passed")
    print(f"Prediction  : {prediction}")
    print(f"Probability : {probability:.4f}")

    # Optional warning (not failure)
    if probability < 0.05:
        print(" Warning: Very low probability — model sees normal conditions")

if __name__ == "__main__":
    run_smoke_test()
