import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from streamlit_cropper import st_cropper

st.title("Face Mask Detection Project")

@st.cache_resource
def load_face_model():
    return load_model("face-mask-detector.keras")

model = load_face_model()


if "cropped_image_data" not in st.session_state:
    st.session_state.cropped_image_data = None

option = st.selectbox("Select", ["Image", "Capture"])
uploaded_file = None
camera_image = None

if option == "Image":
    uploaded_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.write("Crop the image")
        cropped_img = st_cropper(
            image,
            realtime_update=True,
            box_color='red',
            aspect_ratio=(1, 1)
        )
        st.session_state.cropped_image_data = cropped_img.resize((150, 150))
        st.image(st.session_state.cropped_image_data, caption="Processed Image")
else:
    camera_image = st.camera_input("Capture a photo")
    if camera_image:
        image = Image.open(camera_image)
        st.write("Crop the image")
        cropped_img = st_cropper(
            image,
            realtime_update=True,
            box_color='red',
            aspect_ratio=(1, 1)
        )
        st.session_state.cropped_image_data = cropped_img.resize((150, 150))
        st.image(st.session_state.cropped_image_data, caption="Processed Image")


if st.button("Detect"):
    image_to_detect = st.session_state.cropped_image_data
    
    if image_to_detect is not None:     
    
        image_to_detect = image_to_detect.convert("RGB") 
        img_array = np.array(image_to_detect, dtype=np.float32)
        
        
        img_array = (img_array / 127.5) - 1.0
        img_array = np.expand_dims(img_array, axis=0)

        result = model.predict(img_array)
        score = result[0, 0]
        
        st.write("---")
        
        if score > 0.5:
            st.success("🎯 Result: Person is WITH Mask")
        else:
            st.error("🚨 Result: Person is WITHOUT Mask")
    else:
        st.error("Please upload or capture an image first and crop it!")
