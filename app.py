import streamlit as st
import numpy as np
from PIL import Image
import os
import struct
from collections import Counter

# Define the base path where data will be stored on Streamlit Cloud
# This assumes your .npy files are in the same directory as app.py in your GitHub repo.
data_path = '.' # Current directory of app.py

# --- Custom KNN Functions (copied from your notebook) ---
def minkowski_distance(row1, row2, p=2):
    return np.sum(np.abs(row1 - row2)**p)**(1/p)

def get_neighbors(train_data, train_labels, test_row, k, p_value=2):
    distances = []
    for i, train_row in enumerate(train_data):
        dist = minkowski_distance(test_row, train_row, p=p_value)
        distances.append((train_row, train_labels[i], dist))
    distances.sort(key=lambda x: x[2])
    neighbors = distances[:k]
    return neighbors

def predict_classification(train_data, train_labels, test_row, k, p_value=2):
    neighbors = get_neighbors(train_data, train_labels, test_row, k, p_value=p_value)
    neighbor_labels = [neighbor[1] for neighbor in neighbors]
    most_common = Counter(neighbor_labels).most_common(1)
    return most_common[0][0]

# --- Load Refined Training Data ---
@st.cache_resource # Cache the data loading for performance
def load_refined_data(train_size):
    # Adjust these paths if your files are in a subdirectory within the GitHub repo
    refined_images_path = os.path.join(data_path, 'refined_train_images_10k.npy')
    refined_labels_path = os.path.join(data_path, 'refined_train_labels_10k.npy')

    if not os.path.exists(refined_images_path) or not os.path.exists(refined_labels_path):
        st.error(f"Error: Training data not found. Please ensure '{os.path.basename(refined_images_path)}' and '{os.path.basename(refined_labels_path)}' are in the same directory as app.py or adjust paths.")
        st.stop()

    full_train_images = np.load(refined_images_path)
    full_train_labels = np.load(refined_labels_path)

    # Select a subset of the training data based on train_size
    if train_size > len(full_train_images):
        st.warning(f"Requested training size ({train_size}) is larger than available data ({len(full_train_images)}). Using all available data.")
        current_train_images_flat = full_train_images.reshape(len(full_train_images), -1)
        current_train_labels = full_train_labels
    else:
        current_train_images_flat = full_train_images[:train_size].reshape(train_size, -1)
        current_train_labels = full_train_labels[:train_size]

    return current_train_images_flat, current_train_labels

# --- Streamlit App --- 
st.title('Handwritten Digit Recognition using Custom KNN')
st.write('Upload a handwritten digit image (0-9) and the model will predict it.')

# Hyperparameters for the app
APP_K_VALUE = 5
APP_MINKOWSKI_P = 3
APP_TRAINING_SIZE = 500

st.sidebar.header('Model Parameters')
st.sidebar.write(f"**K Value:** {APP_K_VALUE}")
st.sidebar.write(f"**Minkowski p:** {APP_MINKOWSKI_P}")
st.sidebar.write(f"**Training Data Size:** {APP_TRAINING_SIZE}")

# Load the training data once
train_images_flat, train_labels = load_refined_data(APP_TRAINING_SIZE)
st.sidebar.write(f"Loaded training data shape: {train_images_flat.shape}")

uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('L') # Convert to grayscale
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("Processing image...")

    # Preprocess the image
    image = image.resize((28, 28), Image.Resampling.LANCZOS)
    image_array = np.array(image)

    # Invert colors if necessary (MNIST is white digit on black background)
    if np.mean(image_array) > 127: # Heuristic for white background
        image_array = 255 - image_array # Invert colors

    processed_image_flat = image_array.reshape(1, -1)[0]

    st.image(processed_image_flat.reshape(28, 28), caption='Processed for Prediction', width=100)

    # Perform prediction
    with st.spinner('Predicting...'):
        prediction = predict_classification(train_images_flat, train_labels, processed_image_flat, APP_K_VALUE, p_value=APP_MINKOWSKI_P)
        st.success('Prediction complete!')

    st.write(f"## Predicted Digit: **{prediction}**")

else:
    st.write("Please upload an image to start the prediction.")

