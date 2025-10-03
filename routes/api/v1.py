from typing import List
import app.schemas as schemas
from fastapi import APIRouter, Form
from fastapi import UploadFile, File

from app.controllers.ObjectaxController import objectax_controller
from core.dummy.SimulateQueueController import simulateQueueController
from app.controllers.FakturController import fakturautoeval_controller
from app.controllers.SwitchModelController import switch_model_controller
from app.controllers.ItemMatchingController import matching_controller
from app.schemas.AgentMatchingOutputSchema import MatchingSchema

router = APIRouter()

@router.post("/faktur-process")
async def warehouse(files: List[UploadFile] = File(...)):
    return await fakturautoeval_controller.run_workflow(files)

@router.post("/object-process")
async def warehouse(files: List[UploadFile] = File(...), po_number: str = Form(...)):
    return await switch_model_controller.object_model(files, po_number)

@router.post("/simulate")
async def simulate_endpoint(payload: schemas.SimulateItem):
    return await simulateQueueController.add_queue(user_session_id="test_session", input_item=payload)

@router.post("/check")
async def simulate_check_endpoint(payload: schemas.SimulateItem):
    return await simulateQueueController.get_task_status(task_id=payload.text)

@router.post("/matching-process")
async def matching_endpoint(req: MatchingSchema):
    return await matching_controller.match_lists(req)