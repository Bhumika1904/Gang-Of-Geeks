from flask import Flask, render_template, request, redirect, url_for
import os
import torch
from detect_plastic import detect_plastic

# Initialize the Flask app
app = Flask(__name__)

# Directory to save uploaded images
app.config['UPLOAD_FOLDER'] = './static/uploads'

# Route for the homepage
@app.route('/')
def index():
    return render_template('index.html')

# Route for handling file uploads and plastic detection
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(url_for('index'))

    file = request.files['file']

    if file.filename == '':
        return redirect(url_for('index'))

    # Save the uploaded file
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    # Call the function to detect plastic in the image
    prediction = detect_plastic(filepath)

    return render_template('index.html', prediction=prediction, image_url=filepath)

if __name__ == "__main__":
    app.run(debug=True)
