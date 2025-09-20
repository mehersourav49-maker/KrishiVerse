import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# Load the saved model
model = tf.keras.models.load_model("plant_disease_model.keras")

# Class labels (same order as your training data folders)
class_labels = [
    'Apple___healthy', 'Apple___rust', 'Apple___scab',
    'Cherry___healthy', 'Cherry___powdery_mildew',
    'Corn___healthy', 'Corn___northern_leaf_blight',
    'Grape___black_rot', 'Grape___esca',
    'Peach___bacterial_spot', 'Peach___healthy',
    'Potato___early_blight', 'Potato___late_blight',
    'Strawberry___leaf_scorch', 'Tomato___late_blight',
    'Tomato___yellow_leaf_curl'
]

def predict_disease(img_path):
    # Load and preprocess image
    img = image.load_img(img_path, target_size=(224, 224))  # use same size as training
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Predict
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    confidence = np.max(predictions[0])

    print(f"✅ Predicted: {class_labels[predicted_class]} (Confidence: {confidence:.2f})")

# Example usage:
# Change this path to test with your own leaf image
predict_disease("test_leaf.JPG")
