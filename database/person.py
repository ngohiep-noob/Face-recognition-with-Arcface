from database.collection import Collection
from bson.objectid import ObjectId


class Person(Collection):
    def add_person(self, person):
        insert_result = self.insert_one(person)

        return str(insert_result.inserted_id)

    def find_by_id(self, id):
        result = self.find_one({"_id": ObjectId(id)})

        return result
