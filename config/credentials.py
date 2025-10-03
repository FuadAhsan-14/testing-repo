import base64
import json
from google.oauth2 import service_account
from config.setting import env

def google_credential():
    decoded_bytes = base64.b64decode(env.service_account_file + "===")
    decoded_string = decoded_bytes.decode('utf-8')
    credentials = service_account.Credentials.from_service_account_info(json.loads(decoded_string))
    return credentials
