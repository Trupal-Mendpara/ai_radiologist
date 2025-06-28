import tensorflow as tf
import numpy as np
import os
import traceback
import streamlit as st


class_names = ['COVID-19', 'Invalid', 'Normal', 'Pneumonia', 'Tuberculosis']

def predict(image_array):
    model_path = os.path.join(os.path.dirname(__file__), '..', 'disease_classifier.h5')
    model = tf.keras.models.load_model(model_path)
    image_array = np.expand_dims(image_array, axis=0)  # Batch dimension
    prediction = model.predict(image_array)
    class_idx = np.argmax(prediction, axis=1)[0]
    return class_names[class_idx], prediction[0][class_idx]