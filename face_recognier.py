from face_detector import FaceDetector
from face_embedder import FaceEmbedder


class FaceRecognizer:
    def __init__(self) -> None:
        """
        Initialize the face detector and face embedder
        """
        self.face_detector = FaceDetector()
        self.face_embedder = FaceEmbedder()

    def recognize_faces(self, img):
        face = self.face_detector.detect_face(img)  # add cropped face
        face = self.face_embedder.embed_faces(face)  # add embedding

        return face
