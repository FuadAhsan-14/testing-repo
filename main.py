import uvicorn
from app.Kernel import app
# from config.middleware import setup_middleware
from config.routes import setup_routes
from config.exception import setup_exception

# setup_middleware(app)
setup_exception(app)
setup_routes(app)

if __name__ == "__main__":
    uvicorn.run("main:app",
            host="localhost",
            port=8000,
            reload=True
            )
