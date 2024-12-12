# import tensorflow as tf
# import numpy as np
# from tensorflow.keras.preprocessing.image import load_img, img_to_array

# # Load the model
# model = tf.keras.models.load_model("keras_model.h5")

# # Load the labels
# labels = []
# with open("labels.txt", "r") as file:
#     labels = [line.strip() for line in file.readlines()]

# # Load and preprocess the image
# image_path = "image.png"
# image = load_img(image_path, target_size=(224, 224))  # Adjust size if your model uses a different input shape
# image_array = img_to_array(image)
# image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
# image_array = image_array / 255.0  # Normalize the image

# # Predict
# predictions = model.predict(image_array)

# # Print category percentages
# for i, label in enumerate(labels):
#     print(f"{label}: {predictions[0][i] * 100:.2f}%")


from fastapi import FastAPI, File, UploadFile
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from io import BytesIO

# Initialize FastAPI app
app = FastAPI()

# Load the model
model = tf.keras.models.load_model("keras_model.h5")

# Load the labels
labels = []
with open("labels.txt", "r") as file:
    labels = [line.strip() for line in file.readlines()]

# Function to preprocess the image
def preprocess_image(image_file):
    image = load_img(BytesIO(image_file.read()), target_size=(224, 224))  # Adjust size if your model uses a different input shape
    image_array = img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    image_array = image_array / 255.0  # Normalize the image
    return image_array

# API endpoint to classify the image
@app.post("/classify")
async def classify_image(file: UploadFile = File(...)):
    try:
        # Preprocess the uploaded image
        image_array = preprocess_image(file.file)

        # Predict
        predictions = model.predict(image_array)

        # Create a response dictionary
        response = {label: float(predictions[0][i]) for i, label in enumerate(labels)}

        return {"success": True, "predictions": response}

    except Exception as e:
        return {"success": False, "error": str(e)}
