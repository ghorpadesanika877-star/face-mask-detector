import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from streamlit_cropper import st_cropper

st.title("Face Mask Detection Project")

# मॉडेल लोड करण्यासाठी साधे फंक्शन
@st.cache_resource
def load_face_model():
    return load_model("face-mask-detector.keras")

model = load_face_model()

# इमेज इनपुटचे पर्याय
option = st.selectbox("Select", ["Image", "Capture"])
uploaded_file = None
camera_image = None
image_to_detect = None

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
        image_to_detect = cropped_img.resize((150, 150))
        st.image(image_to_detect, caption="Processed Image")
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
        image_to_detect = cropped_img.resize((150, 150))
        st.image(image_to_detect, caption="Processed Image")


if st.button("Detect"):
    if image_to_detect is not None:     
      
        image_to_detect = image_to_detect.convert("RGB") 
        img_array = np.array(image_to_detect, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        

        result = model.predict(img_array)
        
        st.write("---")
        
       
        if result[0, 0] <= 0.6:
            st.success("🎯 Result: Person is WITH Mask")
        else:
            st.error("🚨 Result: Person is WITHOUT Mask")
    else:
        st.error("Please upload or capture an image first!")