import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# 1. Page Configuration
st.set_page_config(page_title="Cat vs Dog Classifier", page_icon="🐾")

# 2. Load the updated model
# Ensure the file name matches exactly what you saved in train.py
@st.cache_resource
def load_my_model():
    return tf.keras.models.load_model('cat_dog_model.keras')

model = load_my_model()

# 3. UI Elements
st.title("🐾 Cat vs Dog Classifier")
st.write("This model uses a Global Average Pooling CNN to keep the file size small and deployment fast.")

uploaded_file = st.file_uploader("Upload an image (JPG, JPEG, or PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image', use_container_width=True)
    
    # 4. Preprocessing
    # We must resize to 128x128 to match the input_shape in train.py
    img_resized = img.resize((128, 128))
    img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalization

    # 5. Prediction Logic
    if st.button("Classify Image"):
        with st.spinner('Analyzing...'):
            prediction = model.predict(img_array)
            
            # Most ImageDataGenerators assign Cat: 0 and Dog: 1
            if prediction[0][0] > 0.5:
                confidence = prediction[0][0] * 100
                st.success(f"It's a **DOG**! 🐶 (Confidence: {confidence:.2f}%)")
            else:
                confidence = (1 - prediction[0][0]) * 100
                st.info(f"It's a **CAT**! 🐱 (Confidence: {confidence:.2f}%)")

# 6. Sidebar Info
st.sidebar.title("About")
st.sidebar.info(
    "This app uses a Convolutional Neural Network (CNN) "
    "trained on the Cats vs Dogs dataset. By using "
    "Global Average Pooling, we've reduced the model size "
    "to be GitHub-friendly!"
)
