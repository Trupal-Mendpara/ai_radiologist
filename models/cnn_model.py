import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model("disease_classifier.h5")

class_names = ['COVID-19', 'Invalid', 'Normal', 'Pneumonia', 'Tuberculosis']

def predict(image_array):
    image_array = np.expand_dims(image_array, axis=0)  # Batch dimension
    prediction = model.predict(image_array)
    class_idx = np.argmax(prediction, axis=1)[0]
    return class_names[class_idx], prediction[0][class_idx]