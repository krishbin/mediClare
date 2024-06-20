from flask import Flask, request, jsonify, render_template, redirect, url_for
import requests
import pytesseract
from PIL import Image
import io
from time import sleep
import markdown2 as md
import csv


app = Flask(__name__)

def extract_text_from_image(image_file):
    img = Image.open(image_file)
    # Perform OCR on the image
    text = pytesseract.image_to_string(img)
    return text

# Serve the HTML template for reports
@app.route('/reports', methods=['GET'])
def reports():
    return render_template('reports.html')

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

# Serve the HTML template for text input
@app.route('/text', methods=['GET'])
def text_simplifier():
    return render_template('text.html')

# Serve the HTML template for chat
@app.route('/chat', methods=['GET'])
def chat():
    rest_api_url = 'http://127.0.0.1:8000/simplify_reset_context/'
    response = requests.get(rest_api_url)
    return render_template('chat.html')

# Endpoint to handle chat interaction
@app.route('/chat-interaction/', methods=['POST'])
def chat_interaction():
    data = request.json
    user_message = data.get('message')

    if not user_message:
        return jsonify({'error': 'No message provided'}), 400

    # Simulate backend model response (replace with actual model integration)
    try:
        # Send extracted text to FastAPI for simplification
        rest_api_url = 'http://127.0.0.1:8000/simplify_text_llm_context/'
        response = requests.get(rest_api_url, json={'input': user_message})

        if response.status_code == 200:
            return jsonify({'response': md.markdown(response.content.decode())})
        else:
            return jsonify({'error': 'Failed to get simplified text'}), response.status_code
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Handle file upload, extract text, and send to FastAPI
@app.route('/upload-pdf/', methods=['POST'])
def upload_pdf():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected for uploading'}), 400

    try:
        # Extract text from image
        extracted_text = extract_text_from_image(file)
        
        # Send extracted text to FastAPI for simplification
        rest_api_url = 'http://127.0.0.1:8000/simplify_text_llm/'
        response = requests.get(rest_api_url, json={'input': extracted_text})

        if response.status_code == 200:
            return jsonify({'simplified_text': md.markdown(response.content.decode())})
        else:
            return jsonify({'error': 'Failed to get simplified text'}), response.status_code
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Handle text input and interaction with REST API
@app.route('/upload-text/', methods=['POST'])
def upload_text():
    data = request.get_json()
    medical_text = data.get('medical_text')

    if not medical_text:
        return jsonify({'error': 'No text provided'}), 400

    try:
        # Replace with the actual URL of your REST API
        rest_api_url = 'http://127.0.0.1:8000/simplify_text_llm/'
        response = requests.get(rest_api_url, json={'input': medical_text})

        if response.status_code == 200:
            return jsonify({'simplified_text': md.markdown(response.content.decode())})
        else:
            return jsonify({'error': 'Failed to get simplified text'}), response.status_code
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Route to display current query-output pair
@app.route('/jargon', methods=['GET'])
def jargon():
    return render_template('jargon.html')
    
if __name__ == '__main__':
    app.run(debug=True,host="0.0.0.0")
