import os
import numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import uvicorn

# Load the trained model
MODEL_PATH = "plant_disease_model.keras"
model = load_model(MODEL_PATH)

# Define class labels (replace with your dataset classes)
class_names = [
    'Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy', 'PlantVillage',
    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
    'Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight',
    'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot',
    'Tomato_Spider_mites_Two_spotted_spider_mite',
    'Tomato__Target_Spot', 'Tomato__Tomato_YellowLeaf__Curl_Virus',
    'Tomato__Tomato_mosaic_virus', 'Tomato_healthy'
]

# Remedies dictionary (matches class_names exactly)
remedies = {
    "Pepper__bell___Bacterial_spot": "Apply copper-based fungicides. Avoid overhead watering.",
    "Pepper__bell___healthy": "No disease detected. Keep maintaining proper care.",
    "PlantVillage": "Dataset placeholder class. Please check your dataset structure.",
    "Potato___Early_blight": "Use resistant varieties, apply fungicides, and rotate crops.",
    "Potato___Late_blight": "Use certified seed potatoes and apply fungicides early.",
    "Potato___healthy": "No disease detected. Maintain good soil health.",
    "Tomato_Bacterial_spot": "Apply copper-based sprays and remove infected leaves.",
    "Tomato_Early_blight": "Use resistant varieties, apply fungicides, rotate crops.",
    "Tomato_Late_blight": "Destroy infected plants, apply fungicides, avoid overcrowding.",
    "Tomato_Leaf_Mold": "Increase air circulation, use resistant varieties, apply fungicides.",
    "Tomato_Septoria_leaf_spot": "Remove infected leaves, rotate crops, apply fungicides.",
    "Tomato_Spider_mites_Two_spotted_spider_mite": "Use miticides, introduce natural predators, spray neem oil.",
    "Tomato__Target_Spot": "Apply fungicides and avoid overhead irrigation.",
    "Tomato__Tomato_YellowLeaf__Curl_Virus": "Control whiteflies, use resistant varieties.",
    "Tomato__Tomato_mosaic_virus": "Remove infected plants, disinfect tools, control weeds.",
    "Tomato_healthy": "No disease detected. Maintain proper nutrition and care."
}

# FastAPI app
app = FastAPI()

# Path to frontend
frontend_path = os.path.join(os.path.dirname(__file__), "frontend")

# Serve static files (CSS, JS, etc.)
app.mount("/static", StaticFiles(directory=frontend_path), name="static")

# Route for frontend
@app.get("/")
async def serve_home():
    file_path = os.path.join(frontend_path, "index.html")
    with open(file_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

# Prediction route
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        with open("temp.jpg", "wb") as f:
            f.write(contents)

        # Preprocess image
        img = image.load_img("temp.jpg", target_size=(224, 224))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Prediction
        preds = model.predict(img_array)
        class_index = np.argmax(preds)
        confidence = float(np.max(preds))
        disease = class_names[class_index]

        # Remedy lookup
        remedy = remedies.get(disease, "No specific remedy available.")

        return JSONResponse({
            "disease": disease,
            "confidence": confidence,
            "remedy": remedy
        })

    except Exception as e:
        return JSONResponse({"error": str(e)})

# Run the server
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
