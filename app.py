from flask import Flask, render_template, request, jsonify, abort
import logging
import base64
from ocr import LicensePlateReader
from v8 import ANPR_Cropper
import cv2
import os
import json
import numpy as np

app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG)  # Set the desired log level

license_plate_reader = LicensePlateReader()
anpr_cropper = ANPR_Cropper(model_path='./models/anpr_v8.pt')

# Route for the HTML template
@app.route('/')
def index():
    return render_template('index.html')

# Route for processing the uploaded image
@app.route('/v1/process-image', methods=['POST'])
def process_image():
    # Get the uploaded file from the POST request
    file = request.files['image']
    if not file.mimetype.startswith('image/'):
        abort(400, "No image file uploaded")
    logging.debug(file)
    logging.debug(type(file))
    if file:
        # Save the uploaded image
        image_path = os.path.join('uploads', file.filename)
        file.save(image_path)
        
        # Read the uploaded image
        image = cv2.imread(image_path)
        
        # Crop the license plate from the image
        cropped_plate = anpr_cropper.crop_license_plate(image)
        
        if cropped_plate is not None:
            # Read the license plate text from the cropped plate
            license_plate_text = license_plate_reader.read_license_plate(cropped_plate)
            
            if license_plate_text:
                # Return the recognized license plate as JSON response
                response = {'results': [{'plate': license_plate_text}]}
                return json.dumps(response)
    
    # If no license plate is found or text cannot be recognized
    response = {'results': [{'plate': 'No license plate found or unable to recognize text.'}]}
    logging.debug(response)
    return json.dumps(response)

# Route for processing the uploaded image
@app.route('/v1/process-base64', methods=['POST'])
def process_base64():
    # Get the base64-encoded image from the POST request
    payload = request.get_json()
    image_b64 = payload['image']
    
    # Decode the base64-encoded image data
    try:
        if "base64," in image_b64:
            image_data = base64.b64decode(str(image_b64.split("base64,")[1].replace(" ", "+")))
        else:
            image_data = base64.b64decode(image_b64)
    except:
        abort(400, "Invalid base64 string")
    
    # Convert the image data to OpenCV format
    nparr = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Crop the license plate from the image
    cropped_plate = anpr_cropper.crop_license_plate(image)
    
    if cropped_plate is not None:
        # Read the license plate text from the cropped plate
        license_plate_text = license_plate_reader.read_license_plate(cropped_plate)
        
        if license_plate_text:
            # Return the recognized license plate as JSON response
            response = {'results': [{'plate': license_plate_text}]}
            return json.dumps(response)
    
    # If no license plate is found or text cannot be recognized
    response = {'results': [{'plate': 'No license plate found or unable to recognize text.'}]}
    logging.debug(response)
    return json.dumps(response)

if __name__ == '__main__':
    app.run(debug=True, port=80, host="0.0.0.0")
