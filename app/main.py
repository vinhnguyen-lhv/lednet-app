import streamlit as st
import cv2
import numpy as np
from PIL import Image

# Set the title of the app
st.title("Image Processing Mockup App")

# Add an image uploader
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read the uploaded image
    image = Image.open(uploaded_file)
    st.write("Original Image")
    
    # Convert the image to an array for processing
    image_array = np.array(image)
    
    # Display the original image
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Process the image (convert to grayscale for this example)
    gray_image = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
    
    # Convert the grayscale image back to PIL format to display
    processed_image = Image.fromarray(gray_image)
    
    # Display the processed image
    st.write("Processed Image")
    st.image(processed_image, caption="Processed Image (Grayscale)", use_column_width=True)
