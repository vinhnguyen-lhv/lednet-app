import streamlit as st
import cv2
import numpy as np
from lednet_utils import lednet_inference

# Set the title of the app
st.title("Image Processing with LEDNet")

# Add an image uploader
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read the uploaded image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    
    # Decode the byte array to an OpenCV image
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    # Process the image (convert to grayscale for this example)
    output_image = lednet_inference(image)
    
    
    # Create two columns for side-by-side display
    col1, col2 = st.columns(2)
    
    # Display the original image in the first column
    with col1:
        st.image(image, channels="BGR", caption="Uploaded Image", use_container_width=True)
    
    # Display the processed image in the second column
    with col2:
        st.image(output_image, channels="BGR", caption="Processed Image (LEDNet)", use_container_width=True)