from flask import Flask, request, jsonify
import requests
import json
import os
app = Flask(__name__)
# IBM Watson Assistant Credentials
API_KEY = "your_ibm_api_key"
ASSISTANT_ID = "your_assistant_id"
URL = "your_ibm_assistant_url"
def ask_watson(message):
    headers = {
        "Content-Type": "application/json"
    }
    data = {
        "input": {
            "text": message
        }
    }
    response = requests.post(
        f"{URL}/v2/assistants/{ASSISTANT_ID}/message?version=2021-06-14",
        headers=headers,
        auth=("apikey", API_KEY),
        json=data
    )
    return response.json()
@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message")
    watson_response = ask_watson(user_input)
    try:
        reply = watson_response["output"]["generic"][0]["text"]
    except:
        reply = "Sorry, I didn’t understand that."
    return jsonify({"reply": reply})

if __name__ == "__main__":
    app.run(debug=True)
