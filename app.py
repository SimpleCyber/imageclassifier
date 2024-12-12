import os
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import tensorflow as tf
from io import BytesIO
import uvicorn

# Create FastAPI app
app = FastAPI()

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Load the Keras model
# Replace 'keras_model.h5' with your actual model filename
model = tf.keras.models.load_model('keras_model.h5')

# Load class labels
# Create a labels.txt file with your class labels, one per line
with open('labels.txt', 'r') as f:
    CLASS_LABELS = [line.strip() for line in f.readlines()]

def preprocess_image(image):
    """
    Preprocess the uploaded image to match Teachable Machine model input requirements
    
    Typical Teachable Machine preprocessing:
    1. Resize to model's expected input size (usually 224x224)
    2. Convert to RGB
    3. Normalize pixel values
    """
    # Resize image to match model's expected input size (modify as needed)
    image = image.resize((224, 224))
    
    # Convert to numpy array and normalize
    img_array = np.asarray(image) / 255.0
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    """
    Endpoint to predict image classification
    """
    # Read the uploaded image file
    contents = await file.read()
    image = Image.open(BytesIO(contents))
    
    # Preprocess the image
    processed_image = preprocess_image(image)
    
    # Make prediction
    predictions = model.predict(processed_image)[0]
    
    # Create results dictionary
    results = {
        label: float(prob) * 100 
        for label, prob in zip(CLASS_LABELS, predictions)
    }
    
    return results

# Optional: Health check endpoint
@app.get("/")
def health_check():
    return {"status": "healthy", "model": "Teachable Machine Classifier"}

# For local testing
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)