from langgraph.checkpoint.mongodb import AsyncMongoDBSaver
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
import asyncio

class MongoDb:
    def __init__(self, conn_string="mongodb://localhost:27017/"):
        try:
            self.client = MongoClient(conn_string)
            self.client.admin.command('ismaster')
            self.db = self.client["research-hope"]

        except ConnectionFailure as e:
            print(f"Could not connect to MongoDB: {e}")
            self.client = None
            self.db = None

    def _sync_find_document(self, collection_name, field_name, value, projection):
        """
        Private, synchronous method with the actual blocking code.
        """
        collection = self.db[collection_name]
        cursor = collection.find({field_name: value}, projection)
        return list(cursor)

    async def find_document_by_field_value(
        self,
        collection_name: str,
        field_name: str,
        value: any,
        projection: dict = None
    ):
        """
        Public async method that safely runs the blocking code in a separate thread.
        """
        if self.db is None:
            print("Database not connected.")
            return []
            
        try:
            result = await asyncio.to_thread(
                self._sync_find_document,
                collection_name,
                field_name,
                value,
                projection
            )
            return result
        except Exception as e:
            print(f"An error occurred: {e}")
            return []

    def close_connection(self):
        if self.client:
            self.client.close()

db = MongoDb()
