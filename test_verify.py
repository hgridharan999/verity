import requests
import json

url = "http://localhost:8000/api/v1/verify"
files = {"file": open("test_image1.png", "rb")}
data = {
    "priority": "standard",
    "include_detailed_report": "true"
}

print("Sending verification request...")
response = requests.post(url, files=files, data=data)

print(f"\nStatus Code: {response.status_code}")
print(f"\nResponse:")
print(json.dumps(response.json(), indent=2))
