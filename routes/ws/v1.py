
# from fastapi import WebSocket, APIRouter
# from app.controllers.SpeechToTextController import speechToTextController

# router = APIRouter()

# @router.websocket("/{client_id}")
# async def websocket_endpoint(websocket: WebSocket, client_id: str, locale: str = 'id'):
#     await speechToTextController.stt_google_v1(websocket, client_id, locale)