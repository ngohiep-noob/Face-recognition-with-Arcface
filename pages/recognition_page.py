import streamlit as st
from PIL import Image
import numpy as np
from app import App
import pymongo

app = App()
st.markdown("# Face Recognition")
st.sidebar.markdown("# Face Recognition")

img_file_buffer = st.camera_input("Take a photo to recognize")

submit_button = st.button("Submit")

if submit_button:
  if img_file_buffer is not None:
          # To read the image file buffer as a PIL Image:
          
          img_buffer = Image.open(img_file_buffer)

          # To convert PIL Image to a numpy array:
          img_array = np.array(img_buffer)
          pid, score = app.recognize(img_array)
          person = app.get_person_info(pid)
          

