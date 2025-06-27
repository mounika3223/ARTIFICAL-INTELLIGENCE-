from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# Load the model only ONCE when the application starts
# This is crucial for performance and avoiding timeouts
model = load_model('chicken_disease_model.h5')

# Update to your class names
class_names = ['Coccidiosis', 'Healthy', 'Newcastle Disease', 'Salmonella']

UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded", 400

    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    img = image.load_img(filepath, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0]
    predicted_index = np.argmax(prediction)
    predicted_class = class_names[predicted_index]
    confidence = prediction[predicted_index]

    disease_info = {
        "Coccidiosis": {
            "medicine": "Amprolium or Toltrazuril",
            "precaution": "Keep litter dry, sanitize equipment, vaccinate chicks."
        },
        "Healthy": {
            "medicine": "No medication needed",
            "precaution": "Continue balanced diet and clean environment."
        },
        "Newcastle Disease": {
            "medicine": "No cure; supportive therapy like multivitamins",
            "precaution": "Vaccinate healthy birds, isolate infected birds, disinfect housing."
        },
        "Salmonella": {
            "medicine": "Antibiotics like Enrofloxacin or Streptomycin",
            "precaution": "Keep feed and water clean, cull carriers, disinfect thoroughly."
        }
    }

    medicine = disease_info[predicted_class]["medicine"]
    precaution = disease_info[predicted_class]["precaution"]

    return render_template(
        'index.html',
        prediction=f"{predicted_class} ({confidence:.2%})",
        medicine=medicine,
        precaution=precaution,
        image_path=filepath,
        labels=class_names,
        confidences=[float(c) for c in prediction]
    )

if __name__ == '__main__':
    # Get the port from the environment variable provided by Render
    port = int(os.environ.get("PORT", 5000)) # Default to 5000 if not set (for local dev)
    app.run(host='0.0.0.0', port=port, debug=False) # Set debug to False for production