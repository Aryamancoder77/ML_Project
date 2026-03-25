from flask import Flask, request, jsonify
from utils import predict

app = Flask(__name__)

@app.route('/')
def home():
    return "Fake News Detection API"

@app.route('/predict', methods=['POST'])
def predict_news():
    data = request.json
    text = data.get("text")
    
    result = predict(text)
    return jsonify({"prediction": result})

if __name__ == '__main__':
    app.run(debug=True)
