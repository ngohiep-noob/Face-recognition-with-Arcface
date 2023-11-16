from database.collection import Collection
from bson.objectid import ObjectId


class Person(Collection):
    def add_user(self, user):
        self.insert_one(user)

    def find_by_id(self, id):
        self.find_one({"_id": ObjectId(id)})
