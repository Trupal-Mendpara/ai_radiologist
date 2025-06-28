import tensorflow as tf
import numpy as np
import os
import traceback
import streamlit as st

class_names = ['COVID-19', 'Invalid', 'Normal', 'Pneumonia', 'Tuberculosis']
@st.cache_resource(show_spinner="Loading model...")
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), '..', 'disease_classifier.h5')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    return tf.keras.models.load_model(model_path)

def predict(image_array):
    if not hasattr(predict, "model"):
        predict.model = load_model()
    image_array = np.expand_dims(image_array, axis=0)  # Batch dimension
    prediction = model.predict(image_array)
    class_idx = np.argmax(prediction, axis=1)[0]
    return class_names[class_idx], prediction[0][class_idx]