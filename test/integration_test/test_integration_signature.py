# import pytest
# import json
# import sys
# import os
# from datetime import datetime, timezone

# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
# from app.utils.SignatureUtils import SignatureUtils

# TEST_SECRET = "test_secret_key"

# @pytest.fixture(autouse=True)
# def mock_env(monkeypatch):
#     class MockEnv:
#         signature_secret = "test_secret_key"
#     monkeypatch.setattr('app.utils.SignatureUtils.env', MockEnv())

# # --- Test Suite ---
# class TestSignatureUtils:
#     """
#     Test suite for the SignatureUtils class. Focuses on the class logic in isolation.
#     """
    
#     def test_verify_signature_valid(self):
#         """
#         Tests that a correctly generated signature is verified successfully.
#         """
#         data = {"message": "hello world", "status": "ok"}
#         timestamp = str(int(datetime.now(timezone.utc).timestamp()))
#         signature = SignatureUtils.create_signature(data, timestamp)
#         is_valid = SignatureUtils.verify_signature(data, timestamp, signature)
#         assert is_valid is True

#     def test_verify_signature_tampered_data(self):
#         """
#         Tests that verification fails if the data has been tampered with.
#         """
#         original_data = {"admin": False, "user": "test"}
#         timestamp = "1678886400"
#         signature = SignatureUtils.create_signature(original_data, timestamp)
#         # Modify the data after signature creation
#         tampered_data = {"admin": True, "user": "test"}
#         is_valid = SignatureUtils.verify_signature(tampered_data, timestamp, signature)
#         assert is_valid is False

#     def test_signature_consistency_with_different_key_order(self):
#         """
#         Tests that data with different key order produces the same signature.
#         """
#         timestamp = "1678886500"
#         data1 = {"name": "Alice", "id": 101}
#         data2 = {"id": 101, "name": "Alice"}
#         signature1 = SignatureUtils.create_signature(data1, timestamp)
#         signature2 = SignatureUtils.create_signature(data2, timestamp)
#         assert signature1 == signature2
    
#     def test_create_signature_with_non_serializable_data(self):
#         """
#         Tests that the function raises a TypeError for non-JSON-serializable data.
#         """
#         data = {"time": datetime.now()}
#         with pytest.raises(TypeError):
#             SignatureUtils.create_signature(data, "1678888000")
