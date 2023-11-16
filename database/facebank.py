from database.collection import Collection


class Facebank(Collection):
    def __init__(self, name, database):
        super().__init__(name, database)

        self.embedding_path = "embedding"
        self.num_candidates = 10
        self.limit = 10

    def add_face(self, image, person_id, embedding):
        face = {
            "image": image,
            "person_id": person_id,
            "embedding": embedding,
        }

        self.insert_one(face)

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
