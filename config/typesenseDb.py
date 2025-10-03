import time
import typesense
from tqdm import tqdm
from .setting import env
from typing import List, Union
from app.utils.Http import HttpClient

class TypesenseDB:
    """
    A class for interacting with Typesense database.

    This class provides functionality to create, delete, and manage collections in Typesense,
    as well as index documents with vector embeddings.

    Attributes:
        client: Typesense client instance for connecting to the database
        collection: Current active collection being operated on
        embed_model: Model name to use for generating embeddings

    Methods:
        multi_search: Property that returns the multi_search client
        validate_collection_exists: Validates if a collection exists
        check_if_collection_exists: Checks if a named collection exists
        create_collection: Creates a new collection with specified schema
        delete_collection: Deletes an existing collection
        index_documents: Indexes documents into a collection with optional embeddings
    """    
    def __init__(self, existing_collection = None, embed_model = None, config=None) -> None:
        config = config or {
            "nodes": [{
                "host": env.typesense_host,
                "port": env.typesense_port,
                "protocol": env.typesense_protocol,
                **({"path": env.typesense_path} if env.typesense_path else {})
            }],
            "api_key": env.typesense_api_key,
            "connection_timeout_seconds": 60,
        }
        self.client = typesense.Client(config)
        self.collection = None
        self.embed_model = embed_model
        if existing_collection:
            if not self.check_if_collection_exists(existing_collection):
                raise Exception(f"Collection {existing_collection} does not exist")
            self.collection = self.client.collections[existing_collection]
        
    def multi_search(self, search_parameters: dict, return_raw:bool = False, exclude_vector=True):
        results = self.client.multi_search.perform(search_parameters, {})
        hits    = results["results"][0].get("hits", [])
        if return_raw:
            return results
        if exclude_vector:
            return [{k: v for k, v in hit["document"].items() if k != "vector"} for hit in hits]
        return [hit['document'] for hit in hits]
      
    def validate_collection_exists(self, collection_name: str = None):
        """
        Validate if a collection exists in the Typesense database.

        Args:
            collection_name (str, optional): Name of the collection to validate.
                If not provided, uses the collection set during initialization.

        Returns:
            Collection: The validated collection object

        Raises:
            Exception: If no collection name is provided and no collection is set during initialization
        """
        if not collection_name and not self.collection:
            raise Exception("Collection name is required Either in initialization or function")
        return self.client.collections[collection_name] if collection_name else self.collection
    
    def check_if_collection_exists(self, collection_name: str) -> bool:
        """
        Check if a collection exists in the Typesense database.

        Args:
            collection_name (str): Name of the collection to check

        Returns:
            bool: True if collection exists, False otherwise

        Raises:
            typesense.exceptions.ObjectNotFound: If collection is not found
        """        
        try:
            self.client.collections[collection_name].retrieve()
            return True
        except typesense.exceptions.ObjectNotFound:
            return False
        
    def create_collection(self, collection_name: str, fields: List[dict], force_recreate:bool, **kwargs) -> None:
        """
        Create a new collection in Typesense with the specified schema.

        Args:
            collection_name (str): Name of the collection to create
            fields (List[dict]): List of field definitions for the collection schema
            force_recreate (bool): If True, delete and recreate collection if it exists
            **kwargs: Additional keyword arguments for the collection schema

        Raises:
            Exception: If an error occurs during collection creation
        """
        if self.check_if_collection_exists(collection_name) and not force_recreate:
            raise Exception(f"Collection {collection_name} already exists")
        self.delete_collection(collection_name)

        try:
            schema = {
                "name": collection_name,
                "fields": fields,
                **kwargs
            }
            self.client.collections.create(schema)
        except Exception as e:
            return str(e)
        
    def delete_collection(self, collection_name: str = None) -> None:
        """
        Delete a collection from the Typesense database.

        Args:
            collection_name (str, optional): Name of the collection to delete. 
                If not provided, uses the collection set during initialization.

        Returns:
            str: Error message if deletion fails, None if successful

        Raises:
            Exception: If collection validation fails or deleting process fails
        """        
        try:
            collection = self.validate_collection_exists(collection_name)
            collection.delete()
        except typesense.exceptions.ObjectNotFound:
            pass
        except Exception as e:
            return str(e)
        
    def index_documents(self, docs: List[dict], embed_column:Union[list[str], str] = None, embed_model:str = None, collection_name:str = None, batch_size:int = 100):
        """
        Index documents into a collection with optional embeddings.

        Args:
            docs (List[dict]): List of documents to index
            embed_column (Union[list[str], str], optional): Column(s) to embed
            embed_model (str, optional): Model name to use for generating embeddings
            collection_name (str, optional): Name of the collection to index documents into.
                If not provided, uses the collection set during initialization.

        Raises:
            Exception: If embed column is not found in the document or if an error occurs during indexing
        """
        collection = self.validate_collection_exists(collection_name)
        current_embed_model = embed_model or self.embed_model
        
        cols_to_embed = []
    
        if current_embed_model:
            if not embed_column:
                raise Exception("Embed column is required for embedding")
            
            if isinstance(embed_column, str):
                cols_to_embed = [col.strip() for col in embed_column.split(',')]
            else:
                cols_to_embed = embed_column

        for doc in tqdm(docs, desc=f"Indexing to '{collection_name}'"):
            if current_embed_model:
                # Process each document to add a vector
                for col in cols_to_embed:
                    if col not in doc:
                        # Skip this doc or raise error, depending on desired behavior
                        print(f"Warning: Embed column '{col}' not found in doc ID {doc.get('id')}. Skipping embedding for this doc.")
                        continue

                embed_text = " ".join(
                    [
                        str(doc.get(col, '')).lower() for col in cols_to_embed
                    ]
                ).strip()

                if embed_text:
                    emb = HttpClient.get_embedding_from_api(f"{env.base_url_embed}{current_embed_model}", embed_text)
                    doc['vector'] = emb
            
            try:
                collection.documents.upsert(doc)
            except Exception as e:
                print(f"❌ Error indexing document: {e}")
                
    def is_alive(self) -> bool:
        """Return True if the Typesense server is reachable and healthy."""
        try:
            res = self.client.health.retrieve()  # {"ok": True} when healthy
            return bool(res.get("ok", False))
        except Exception:
            return False

    def assert_alive(self) -> None:
        """Raise if the Typesense server is not reachable/healthy."""
        if not self.is_alive():
            raise RuntimeError("Typesense server is not reachable or unhealthy. "
                               "Check host, port, protocol, path, and API key.")

    def ensure_collection(self, collection_name: str) -> None:
        """Raise if the given collection does not exist."""
        self.assert_alive()
        if not self.check_if_collection_exists(collection_name):
            raise RuntimeError(f"Collection '{collection_name}' does not exist.")

    def wait_until_ready(self, timeout: int = 15, interval: float = 0.5) -> bool:
        """Poll health until the server is ready or timeout; returns True if ready."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.is_alive():
                return True
            time.sleep(interval)
        return False
    
db = TypesenseDB()
