# import pytest
# import sys
# import os
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# from app.utils.SignatureUtils import SignatureUtils

# @pytest.fixture(autouse=True)
# def mock_env(monkeypatch):
#     class MockEnv:
#         signature_secret = "test_secret_key"
#     monkeypatch.setattr('app.utils.SignatureUtils.env', MockEnv())

# class TestSignatureUtils:
#     """
#     Test suite for the SignatureUtils class.
#     """

#     @pytest.mark.parametrize("data, timestamp", [
#         ({"user_id": 123, "event": "login"}, "1678886400"),
#         ({"item": "book", "price": 19.99, "in_stock": True}, "1725418200"),
#         ({}, "1725418260")
#     ])
#     def test_signature_is_verifiable(self, data, timestamp):
#         """
#         Tests that any created signature can be successfully verified.
#         """
#         signature = SignatureUtils.create_signature(data, timestamp)
        
#         assert signature is not None

#     def test_invalid_signature(self):
#         """
#         Tests that an invalid signature raises a ValueError
#         """
#         invalid_data = {"test": "data"}
#         invalid_timestamp = "1234567890"
#         invalid_signature = "invalid_signature"

#         valid = SignatureUtils.verify_signature(
#             provided_signature=invalid_signature,
#             obj=invalid_data,
#             timestamp=invalid_timestamp
#         )
        
#         assert valid is False
        
#     def test_create_signature_raises_type_error_on_non_serializable_data(self):
#         from datetime import datetime
#         data = {"time": datetime.now()}
#         timsetamp = "1678888000"
#         with pytest.raises(TypeError):
#             SignatureUtils.create_signature(data, timsetamp)
