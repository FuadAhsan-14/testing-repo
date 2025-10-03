from phoenix.otel import register
from config.setting import env
import requests

class Phoenix:
    
    @staticmethod
    def init():
        if ( hasattr(env, 'phoenix_endpoint') 
            and hasattr(env, 'phoenix_api_key') 
            and env.phoenix_endpoint 
            and env.phoenix_api_key
            ):
            is_reachable = False
        try:
            response = requests.get(env.phoenix_endpoint, timeout=3)
            if response.status_code < 500: # Any non-server-error response
                is_reachable = True
            else:
                print(f"⚠️ Phoenix endpoint returned a server error: {response.status_code}.")

        except requests.exceptions.RequestException as e:
            print(f"❌ Phoenix endpoint is not reachable. Tracing disabled. Error: {type(e).__name__}")

        if is_reachable:
            register(
                project_name=env.app_name,
                endpoint=env.phoenix_endpoint,
                auto_instrument=True,
                batch=True,
                verbose=False,
                headers = {"Authorization": f"Bearer {env.phoenix_api_key}"}
            )

    @staticmethod
    def metadata():
        return {
            "service_version": env.app_version,
            "deployment_environment": env.app_env,
        }
