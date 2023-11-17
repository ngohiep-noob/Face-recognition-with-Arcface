import streamlit as st
from PIL import Image
import numpy as np
from app import App
import pymongo


st.markdown("# Face Register")
st.sidebar.markdown("# Face Register")
app = App()

img_file_buffer = st.camera_input("Take a photo to register")

user_type = st.selectbox("Select User Type", ["New User", "Registered User"])
registered_users = app.person_col.get_all()


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
            app.add_new_person(name=new_user_name, image=img_array)
            st.image(img_array, caption=new_user_name)
          elif user_type == "Registered User":
              app.add_new_face(image = img_array, person_id = 1)
              st.image(img_array, caption=selected_user)
