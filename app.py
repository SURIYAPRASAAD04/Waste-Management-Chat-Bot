import os
from flask import Flask, render_template, request, jsonify
from google.cloud import dialogflow_v2 as dialogflow

app = Flask(__name__)

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "new-qfxx-732a273ebbed.json"

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get_response", methods=["POST"])
def get_response_route():
    user_message = request.json.get("message")
    
    dialogflow_session_client = dialogflow.SessionsClient()
    session_id = "Sur@6904"  
    session = dialogflow_session_client.session_path("new-qfxx", session_id)
    text_input = dialogflow.TextInput(text=user_message, language_code="en")
    query_input = dialogflow.QueryInput(text=text_input)
    response = dialogflow_session_client.detect_intent(request={"session": session, "query_input": query_input})
    fulfillment_text = response.query_result.fulfillment_text
    return jsonify({"response": fulfillment_text})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

