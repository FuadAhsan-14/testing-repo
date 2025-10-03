from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # App Info
    app_env: str
    app_name: str
    app_version: str
    
    # Docker
    docker_ports: str
    docker_worker_count: int

    # Security
    jwt_hs_secret: str
    jwt_rs_public_key: str
    jwt_rs_private_key: str
    signature_secret: str
    signature_timeout: int
    scheduler_timezone: str
    allowed_origins: str = "http://localhost:8080"
    allowed_jwt: str 
    
    # Redis (General & Ratelimit)
    redis_host: str
    redis_port: int
    redis_db: int
    redis_password: str
    redis_username: str
    redis_ratelimit_host: str
    redis_ratelimit_port: int
    redis_ratelimit_db: int
    redis_expires_sec: int

    # Database
    db_user: str
    db_password: str
    db_host: str
    db_port: int
    db_name: str

    # Clickhouse
    clickhouse_host: str
    clickhouse_http_port: str
    clickhouse_user: str
    clickhouse_password: str
    clickhouse_database: str
    
    # apm
    apm_server_url: str
    apm_service_name: str
    
    # embed config
    base_url_embed: str
    async_qwen3_embed: str
    
    # Gemini Model
    gemini_regular_model: str = None
    gemini_mini_model: str = None
    gemini_thinking_model: str = None
    gemini_model_pro25:str
    gemini_model_flash15:str
    gemini_model_flash2:str
    gemini_model_flash25:str
    gemini_model_flash_lite25: str
    
    # OpenAI Model
    openai_regular_model: str = None
    openai_mini_model: str = None
    openai_thinking_model: str = None
    gpt_4o_model: str
    gpt_4o_mini_model: str
    o3_mini_model: str
    gpt_5_model: str
    gpt_5_chat_model: str
    gpt_5_mini_model: str
    gpt_5_nano_model: str

    # Claude Models
    claude_model_sonnet_35: str
    claude_model_sonnet_37: str
    claude_model_sonnet_4: str
    claude_model_sonnet_45: str

    # Vertex AI & Google AAPI
    google_api_key: str
    bucket_name: str
    google_project_name: str
    service_account_file: str
    service_account_file_location: str
    
    # Azure API
    azure_api_key: str
    azure_api_key_002: str
    azure_api_key_dev: str
    azure_api_version: str
    azure_api_version_002: str
    azure_api_version_dev: str
    azure_endpoint: str
    azure_endpoint_002: str
    azure_endpoint_dev: str
    
    # AWS API
    aws_region: str
    aws_access_key_id: str
    aws_secret_access_key: str
    aws_region: str
    
    # Typesense
    typesense_api_key: str
    typesense_host: str
    typesense_port: str
    typesense_protocol: str
    typesense_path: str = None
    
    # Typesense Collection Name
    invoice_collection_name: str
    
    # Langsmith
    # langsmith_tracing: str
    # langsmith_api_key: str
    # langsmith_project: str
    # langsmith_endpoint: str
   
   # Other
    phoenix_api_key: str = None
    phoenix_endpoint: str = None
    base_url_uploader: str
    usd_to_idr: float
   
    model_config = SettingsConfigDict(env_file=".env")

env = Settings()

def reload():
    env.__init__()
