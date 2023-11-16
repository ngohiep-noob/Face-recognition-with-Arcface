from database.database import Database
from database.facebank import Facebank
from database.person import Person
from face_detector import FaceDetector
from face_embedder import FaceEmbedder
import cv2


class App:
    def __init__(self) -> None:
        self.database = Database(
            db_url="mongodb+srv://hoanghiephai:3X4qobjRUBpUBcDR@face-embeddings.m9dpaia.mongodb.net/",
            db_name="face_recognition",
        )
        self.face_detector = FaceDetector()
        self.face_embedder = FaceEmbedder()

        self.person_col = Person(name="person", database=self.database)
        self.facebank_col = Facebank(
            name="facebank", database=self.database, embedding_path="face_embedding"
        )

        print("App initialized!")

    def get_embedding(self, image):
        face = self.face_detector.detect_face(image)
        embedding = self.face_embedder.embed_face(face)

        return embedding, face

    def add_new_person(self, name, image):
        embedding, face = self.get_embedding(image)

        person_id = self.person_col.add_person({"name": name})
        self.facebank_col.add_face(image=face, person_id=person_id, embedding=embedding)

        return person_id

    def get_all_people(self):
        return self.person_col.get_all()

    def get_faces_by_person_id(self, person_id):
        return self.facebank_col.get_faces_by_person_id(person_id)

    def add_new_face(self, person_id, image):
        embedding, face = self.get_embedding(image)

        self.facebank_col.add_face(image=face, embedding=embedding, person_id=person_id)

    def recognize_person(self, image):
        pass


if __name__ == "__main__":
    app = App()

    # UNCOMMENT THIS TO ADD NEW PERSON
    # img1 = cv2.imread("sample\hiep-dep-trai.jpg")
    # img2 = cv2.imread("sample\hiep-handsome.jpg")

    # pid = app.add_new_person("Ngo Hiep", img1)
    # app.add_new_face(person_id=pid, image=img2)

    # UNCOMMENT THIS TO GET FACES BY PERSON ID
    # pid = "655624f0bbe1e9caaaab6434"

    # faces = app.get_faces_by_person_id(pid)

    # for face in faces:
    #     cv2.imshow("face", face["image"])

    # cv2.waitKey(0)
