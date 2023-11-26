# app.py
from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
from io import BytesIO
import base64
from PIL import Image



app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/pencil_sketch')
def pencil_sketch():
    return render_template('pencil_sketch.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        # Get user-defined parameters
        blur = int(request.form.get('blur', 5))
        threshold = int(request.form.get('threshold', 128))

        img_str = process_image(file, blur, threshold)

        return jsonify({'image': img_str})

def process_image(file, blur=5, threshold=128):
    image_stream = file.read()
    nparr = np.frombuffer(image_stream, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    inverted_img = 255 - gray
    
    # Apply user-defined options
    blurred_img = cv2.GaussianBlur(inverted_img, (blur * 2 + 1, blur * 2 + 1), 0)
    _, sketch = cv2.threshold(blurred_img, threshold, 255, cv2.THRESH_BINARY_INV)
    
    # Apply Gaussian blur again to the sketch if needed
    final_sketch = cv2.GaussianBlur(sketch, (blur * 2 + 1, blur * 2 + 1), 0)

    # Convert image to base64
    _, buffer = cv2.imencode('.png', final_sketch)
    img_str = base64.b64encode(buffer).decode('utf-8')

    return img_str


if __name__ == '__main__':
    app.run(debug=True)
