import requests


class HttpClient:
    def __init__(self, env):
        self.env = env

    def get(self, env_name:str):
        try:
            res = requests.get(getattr(self.env, env_name))
            return res.json()
        except Exception as e:
            return str(e)
    
    def get_url(self, url: str):
        try:
            res = requests.get(url)
            res.raise_for_status()
            return res.json()
        except requests.exceptions.RequestException as e:
            return {"error": str(e)}
        
    def get_embedding_from_api(embed_api_url: str, text: str) -> list:
        """Mendapatkan embedding dari API eksternal"""
        try:
            response = requests.post(
                embed_api_url,
                json={"query": text},
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()
            result = response.json()

            if result["status"] != 1:
                raise ValueError("Embedding API returned error status")

            return result["data"]["text"]
        except Exception as e:
            print(f"Error getting embedding: {str(e)}")
            raise

    