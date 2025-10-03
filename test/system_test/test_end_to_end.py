# import pytest
# import os 
# import sys
# import json

# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
# from routes.api.v1 import endpoint_test

# @pytest.mark.asyncio
# async def test_simple_endpoint_test():
#     """
#     This is a very simple UNIT TEST. It directly calls the 'test' endpoint
#     function and verifies its output. No server is needed.
#     """
#     result = await endpoint_test(api_version=1)
#     assert result.status_code == 200
    
#     response_data = json.loads(result.body.decode())
#     assert response_data['data'] == {
#         "message": "Input processed",
#         "version": 1,
#         "input": {
#             "text": "Hello"
#         }
#     }
