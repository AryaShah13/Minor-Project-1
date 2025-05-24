import requests

url = "http://192.168.53.238:5001/predict"
image_path = r"C:/Users/ahsha/Desktop/your_image_name.jpg"  # Make sure this is correct

try:
    with open(image_path, "rb") as f:
        files = {"file": f}
        response = requests.post(url, files=files)

        print("Raw Response:", response.text)
        try:
            print("Prediction:", response.json())
        except requests.exceptions.JSONDecodeError:
            print("Error: Response is not in JSON format!")

except FileNotFoundError as e:
    print("Error:", e)
