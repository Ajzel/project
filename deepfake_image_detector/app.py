from flask import Flask, request, render_template
import os
from detection.predict_image import predict_image

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads/'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files.get('image')
    if not file:
        return render_template('result.html', message="❌ No image file uploaded.", score="N/A")

    image_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(image_path)

    score = predict_image(image_path, 'models/deepfake_model.pth')

    if score is None:
        message = "❌ Error during prediction. Please verify model file and image format."
        final_score = "N/A"
    elif score > 0.5:
        message = "⚠️ This image is likely a deepfake."
        final_score = f"{score:.2f}"
    else:
        message = "✅ This image appears authentic."
        final_score = f"{score:.2f}"

    return render_template('result.html', message=message, score=final_score)

if __name__ == '__main__':
    app.run(debug=True)
