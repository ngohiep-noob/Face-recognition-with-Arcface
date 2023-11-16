import streamlit as st
from PIL import Image
import numpy as np


st.markdown("# Face Register")
st.sidebar.markdown("# Face Register")

img_file_buffer = st.camera_input("Take a photo to register")

user_type = st.selectbox("Select User Type", ["New User", "Registered User"])
registered_users = ["User1", "User2", "User3"]

if user_type == "New User":
      new_user_name = st.text_input("Enter New Name:")
elif user_type == "Registered User":
      selected_user = st.selectbox("Select Registered User", registered_users)

submit_button = st.button("Submit")

if submit_button:
      if img_file_buffer is not None:
          # To read the image file buffer as a PIL Image:
          
          img_buffer = Image.open(img_file_buffer)

          # To convert PIL Image to a numpy array:
          img_array = np.array(img_buffer)

          # Display the image with the selected user's name as the caption
          if user_type == "New User":
            st.write("New Face Register successfully")
            st.image(img_array, caption=new_user_name)
          elif user_type == "Registered User":
              st.image(img_array, caption=selected_user)
