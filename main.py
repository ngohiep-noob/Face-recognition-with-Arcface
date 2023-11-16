from face_recognier import FaceRecognizer
from database.person import Person
from database.facebank import Facebank
from database.database import Database

if __name__ == "__main__":
    face_recognizer = FaceRecognizer()
    db = Database(
        db_url="mongodb+srv://hoanghiephai:3X4qobjRUBpUBcDR@face-embeddings.m9dpaia.mongodb.net/",
        db_name="face_recognition",
    )

    person = Person(name="person", database=db)

    facebank = Facebank(name="facebank", database=db)
    # Phrase 1: Add new person
    # 1.1. Add new person to database
    # 1.2. Add new face to facebank(image, person_id, embedding)

    # Phrase 2: Recognize person
    # 2.1. Get face from camera
    # 2.2. Get similar faces from facebank(person_id)
