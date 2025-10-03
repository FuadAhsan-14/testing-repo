# from elasticapm.contrib.starlette import ElasticAPM
from starlette.middleware.cors import CORSMiddleware
# from .logger import apm
from .setting import env
#client = Client()
origins = env.allowed_origins.split(",")

def setup_middleware(app):
    # pass
    app.add_middleware(
        CORSMiddleware, 
        allow_origins=origins, 
        allow_credentials=True, 
        allow_headers=["*"],
        allow_methods=["GET", "POST", "OPTIONS"],
        max_age=600,
    )
    # app.add_middleware(ElasticAPM, client=apm)
