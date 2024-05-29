import requests

def send_symptoms(symptoms):
    url = 'http://localhost:5000/predict'
    headers = {'Content-Type': 'application/json'}

    # Prepare the JSON payload
    data = {'symptoms': symptoms}

    # Make the POST request
    response = requests.post(url, headers=headers, json=data)

    # Handle the response
    if response.status_code == 200:
        predictions = response.json().get('predictions', [])
        if predictions:
            print("Top predicted disorders:")
            for index, disorder in enumerate(predictions, start=1):
                print(f"{index}. {disorder}")
        else:
            print("No predictions returned.")
    else:
        print(f"Error: {response.text}")

if __name__ == '__main__':
    user_input = input("Enter symptoms (comma-separated): ").strip()
    if user_input:
        send_symptoms(user_input)
    else:
        print("No symptoms entered.")
