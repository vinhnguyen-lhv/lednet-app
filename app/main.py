import streamlit as st
import cv2
import numpy as np
from lednet_utils import lednet_inference

# Set the layout to the streamlit app as wide 
st.set_page_config(layout='wide')

# Setup the sidebar
with st.sidebar: 
    st.image('assets/LEDNet_LOLBlur_logo.png')
    st.title('LEDNet Powered App')
    st.info('This application is powered by the LEDNet deep learning model.')

st.title('LEDNet Powered App') 

# Add an image uploader
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read the uploaded image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    
    # Decode the byte array to an OpenCV image
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    # Process the image (Enhance the image)
    output_image = lednet_inference(image)
    
    
    # Create two columns for side-by-side display
    col1, col2 = st.columns(2)
    
    # Display the original image in the first column
    with col1:
        st.image(image, channels="BGR", caption="Uploaded Image", use_container_width=True)
    
    # Display the processed image in the second column
    with col2:
        st.image(output_image, channels="BGR", caption="Processed Image (LEDNet)", use_container_width=True)