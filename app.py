import os
import cv2
import requests
import joblib
import tempfile
from flask import Flask, request, jsonify

app = Flask(__name__)
app.config['ALLOWED_EXTENSIONS'] = set(['jpg', 'png', 'jpeg'])

def allowed_extension(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def check_server_availability(destination_url, timeout=30):
    try:
        response = requests.get(destination_url, timeout=timeout)
        if response.status_code == 400:
            return True
        else:
            return False
    except requests.exceptions.Timeout:
        return False

# Load pre-trained ResNet50 model
def loadmodel():
    model = joblib.load('asset/model/svm_model_with_cv_224.sav')
    return model

def processImage(image, target_size=(224, 224)):
    img = cv2.imread(image)
    img_resized = cv2.resize(img, target_size)
    img_flat = img_resized.flatten()
    img_2d = img_flat.reshape(1, -1)

    return img_2d

def predict_class(image):
    model = loadmodel()
    predictions = model.predict(image)

    return predictions

@app.route("/", methods = ['GET'])
def homepage():
    try:
        # Membuka file HTML
        with open("static/index.html", "r") as file:
            # Membaca dan mengembalikan konten HTML
            return file.read()
    except IOError as e:
        print("Error:", e)
        return None

# @app.route("api/reccomendation", methods = ['POST'])
# def plant_recommendation():
    # Mendapatkan data atribut dari permintaan POST
    # data = request.json
    
    # # Membuat DataFrame dari data yang diterima
    # df = pd.DataFrame(data, index=[0])
    
    # # Membuat prediksi dengan model Decision Tree
    # prediction = model.predict(df)
    
    # # Mengembalikan hasil prediksi
    # return jsonify({"jenis_tanah": prediction[0]})

@app.route("/api/predict", methods = ['POST'])
def soil_prediction():
        image = request.files["image"]
        if image:
            # Membuat file sementara untuk menyimpan file gambar
            temp_dir = tempfile.mkdtemp()
            temp_path = os.path.join(temp_dir, 'temp_image.jpg')
            image.save(temp_path)

            processed_image = processImage(temp_path)
            print(processed_image)
            predicted_class = predict_class(processed_image)
            print(predicted_class)

            os.remove(temp_path)
            os.rmdir(temp_dir)
            return jsonify({
                "data": {
                    "jenis_tanah": predicted_class.tolist()
                }, 
                "status": {
                    "code": 200,
                    "message": "successfully predicted soil type"
                },
            }), 200
        else:
            return jsonify({
                "error": "image file needed"
                }), 400

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8081)))