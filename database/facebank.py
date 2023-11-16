from database.collection import Collection
from utils import decode_image, encode_image


class Facebank(Collection):
    def __init__(self, name, database, embedding_path):
        super().__init__(name, database)

        self.embedding_path = embedding_path
        self.num_candidates = 10
        self.limit = 10

    def add_face(self, image, embedding, person_id):
        encoded_image = encode_image(image)

        face = {
            "image": encoded_image,
            "image_shape": image.shape,
            "person_id": person_id,
            "embedding": embedding.reshape(-1).tolist(),
        }

        self.insert_one(face)

    def get_faces_by_person_id(self, person_id):
        result = self.find({"person_id": person_id})

        for face in result:
            face["image"] = decode_image(
                bytes=face["image"], target_shape=face["image_shape"]
            )

        return result

    def get_similar_face(self, embedding):
        results = self.collection.aggregate(
            [
                {
                    "$vectorSearch": {
                        "queryVector": embedding,
                        "path": self.embedding_path,
                        "numCandidates": self.num_candidates,
                        "limit": self.limit,
                        "index": "default",
                    }
                }
            ]
        )

        return results
