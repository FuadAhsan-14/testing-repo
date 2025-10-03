import json
import hmac
import hashlib
from typing import Any, Union
from datetime import datetime, timezone
from config.setting import env

class SignatureUtils:
    
    @staticmethod
    def create_signature(data: dict = None, timestamp: str = None) -> Union[str, tuple[str, str]]:
        """
        Creates a signature from a dictionary and a timestamp using a canonical JSON string.
        Raises ValueError if the timestamp is empty or invalid.
        Returns signature string if timestamp is provided, otherwise returns tuple of (signature, timestamp)
        """
        if timestamp is None:
            timestamp = str(int(datetime.now(timezone.utc).timestamp()))
        
        if not isinstance(timestamp, str) or not timestamp.strip():
            raise ValueError("Timestamp cannot be empty.")

        key_bytes = env.signature_secret.encode('utf-8')
        
        if data is not None:
            stringified_data_str = json.dumps(data)
            canonical_data = json.loads(stringified_data_str)
            canonical_json = json.dumps(canonical_data, sort_keys=True, separators=(',', ':'))
        else:
            canonical_json = ""
            
        message_to_sign = f"{timestamp}.{canonical_json}".encode('utf-8')
        
        hash_object = hmac.new(key_bytes, message_to_sign, hashlib.sha256)
        signature = hash_object.hexdigest()
        
        if timestamp is not None:
            return signature
        return signature, timestamp

    @staticmethod
    def verify_signature(obj: Any, timestamp: str, provided_signature: str) -> bool:
        """
        Verifies a signature by regenerating it from the raw body and timestamp.
        Returns False if the body is not valid JSON or the timestamp is empty.
        """
        if not isinstance(timestamp, str) or not timestamp.strip():
            return False

        data_to_verify = obj
        if isinstance(obj, bytes):
            try:
                data_to_verify = json.loads(obj.decode('utf-8')) if obj else None
            except json.JSONDecodeError:
                return False
        
        try:
            signature = SignatureUtils.create_signature(data_to_verify, timestamp)
            return hmac.compare_digest(signature, provided_signature)
        except ValueError:
            return False

