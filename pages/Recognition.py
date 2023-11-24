import streamlit as st

from app import App
from utils import draw_bounding_boxes
import cv2

if __name__ == "__main__":
    app = App()
    st.markdown("# Face Recognition App")
    st.sidebar.markdown("# Face Recognition")

    cap = cv2.VideoCapture(0)

    img_holder = None

    while True:
        _, frame = cap.read()

        if frame is None:
            continue

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        identified_faces = app.identify_faces(frame)
        drawn_img = draw_bounding_boxes(frame, identified_faces)

        if img_holder is not None:
            img_holder.empty()

        img_holder = st.image(drawn_img)

        if cv2.waitKey(1) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
