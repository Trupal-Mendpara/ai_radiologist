import tensorflow as tf
import numpy as np
import os
import traceback
import streamlit as st


model_path = os.path.join(os.path.dirname(__file__), '..', 'disease_classifier.h5')

try:
    st.write(f"Loading model from: {model_path}")
    model = tf.keras.models.load_model(model_path)
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error("❌ Error loading model:")
    st.error(str(e))
    st.text(traceback.format_exc())
class_names = ['COVID-19', 'Invalid', 'Normal', 'Pneumonia', 'Tuberculosis']

def predict(image_array):
    image_array = np.expand_dims(image_array, axis=0)  # Batch dimension
    prediction = model.predict(image_array)
    class_idx = np.argmax(prediction, axis=1)[0]
    return class_names[class_idx], prediction[0][class_idx]