from langchain_google_vertexai import (
    VertexAI,
    ChatVertexAI,
    HarmBlockThreshold,
    HarmCategory,
)
from config.setting import env
from config.credentials import google_credential
from langchain_openai import AzureChatOpenAI
from langchain_aws import ChatBedrock
import boto3

SERVICE_ACCOUNT_FILE = env.service_account_file_location
class GenAI:
    def __init__(self):
        self.project = env.google_project_name        
        self.credentials = google_credential()
    
    def ggenai(self, 
            model,
            temperature = 0.0, 
            safety_settings = {
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            },
            **kwargs
        ):
        return VertexAI(
            model_name = model,
            temperature = temperature,
            safety_settings = safety_settings,
            project = self.project,
            max_retries = 1,
            credentials=self.credentials,
            top_p=0.95,
            **kwargs
        )

    def chatGgenai(
        self,
        model: str,
        temperature=0.0,
        safety_settings={
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        },
        **kwargs
    ):
        return ChatVertexAI(
            model_name=model,
            temperature=temperature,
            safety_settings=safety_settings,
            project=self.project,
            max_retries=1,
            credentials=self.credentials,
            **kwargs
        )

    def chatAzureOpenAi(
        self,
        model: str = env.gpt_4o_model,
        deployment: str = "003",
        disable_temperature: bool = False,
        temperature: float = 0.0,
        **kwargs
    ) -> AzureChatOpenAI:
        version_configs = {
            "002": {
                "api_key": env.azure_api_key_002,
                "api_version": env.azure_api_version_002,
                "azure_endpoint": env.azure_endpoint_002,
            },
            "003": {
                "api_key": env.azure_api_key,
                "api_version": env.azure_api_version,
                "azure_endpoint": env.azure_endpoint,
            },
            "dev": {
                "api_key": env.azure_api_key_dev,
                "api_version": env.azure_api_version_dev,
                "azure_endpoint": env.azure_endpoint_dev,
            }
        }

        args = {
            "model": model,
            "temperature": temperature,
            **version_configs.get(deployment, {}),
            **kwargs,
        }

        if disable_temperature or deployment == "dev":
            args.pop("temperature", None)
        return AzureChatOpenAI(**args)
            
    def chatBedrock(
        self,
        model: str = env.claude_model_sonnet_37,
        temperature: float = 0.0,
        region_name: str = env.aws_region,
        aws_access_key_id: str = env.aws_access_key_id,
        aws_secret_access_key: str = env.aws_secret_access_key,
        return_session: bool = False,
        **kwargs
    ) -> ChatBedrock:
        session = boto3.Session(
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=region_name
        )

        return ChatBedrock(
            model_id = model,
            model_kwargs={"temperature": temperature},
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=region_name
        ) if not return_session else session


