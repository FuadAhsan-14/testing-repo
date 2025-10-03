import asyncio
import os
import uuid
import cv2
import numpy as np
from ultralytics import YOLO
import json
from typing import Literal
from urllib.parse import urlparse
from starlette.exceptions import HTTPException
from langgraph.graph import START, StateGraph, END
from google.cloud import storage
from config.setting import env
from config.credentials import google_credential
from app.generative.manager import manager
from app.models.ObjectState import State_ax
from app.services.objectax.PrimaryAgent import PrimaryAgentService
from app.services.objectax.AnomalyAgent import AnomalyAgentService
from app.services.objectax.OrdinaryAgent import OrdinaryBoxAgentService
from app.services.objectax.CardBoxAgent import CardboxAgentService
from app.services.objectax.BnEdAgent import BnedAgentService
from app.utils.Http.HttpResponseUtils import response_success, response_error 
from app.utils.Helper import upload_images_to_gcs

class objectaxController:
    def __init__(self):

        model_path = os.path.abspath("assets/ml_model/yolo12s_dataset7_100e_best.pt")
        self.yolo_model = YOLO(model_path)
        # self.yolo_model = YOLO("assets/ml_model/yolo12s_dataset5_280e_1.pt")
        llm_anomaly = manager._get_llm(name="gemini_model_flash2", temperature=0)
        llm_primary = manager._get_llm(name="gemini_model_flash25", temperature=0)
        llm_cardbox = manager._get_llm(name="claude_model_sonnet_4", temperature=0)
        llm_ordinary = manager._get_llm(name="claude_model_sonnet_4", temperature=0)
        llm_bned = manager._get_llm(name="gemini_model_flash25", temperature=0)

        credentials = google_credential()
        self.gcs_client = storage.Client(project=env.google_project_name, credentials=credentials)
        self.gcs_bucket = self.gcs_client.get_bucket(env.bucket_name)

        self.anomaly_detection = AnomalyAgentService(llm=llm_anomaly)
        self.primary_extraction = PrimaryAgentService(llm=llm_primary)
        self.ordinary_box_extraction = OrdinaryBoxAgentService(llm=llm_ordinary)
        self.cardbox_extraction = CardboxAgentService(llm=llm_cardbox)
        self.bned_extraction = BnedAgentService(llm=llm_bned)

    async def _init_workflow(self):
      builder = StateGraph(State_ax)
          
      # builder.add_node(self.preprocess_crop)
      builder.add_node("crop_and_upload_node", self.crop_and_upload_node)
      builder.add_node("anomaly_detection", self.anomaly_detection)
      builder.add_node("primary_extraction", self.primary_extraction)
      builder.add_node("ordinary_box_extraction", self.ordinary_box_extraction)
      builder.add_node("cardbox_extraction", self.cardbox_extraction)
      builder.add_node("bned_extraction", self.bned_extraction)

      builder.add_edge(START, "crop_and_upload_node") 
      builder.add_edge("crop_and_upload_node", "anomaly_detection") 
      builder.add_conditional_edges("anomaly_detection", self.route_anomaly)
      builder.add_conditional_edges("primary_extraction", self.route_quantity)
      builder.add_edge("primary_extraction", "bned_extraction")
      builder.add_edge("bned_extraction", END)
      builder.add_edge("cardbox_extraction", END)
      builder.add_edge("ordinary_box_extraction", END) 

      compiled_graph = builder.compile()

      return compiled_graph
    
    async def route_anomaly(self, state:State_ax) -> Literal["primary_extraction", END]:
      try:
        if state["anomaly"]["is_anomaly"] == False:
          return "primary_extraction"
        else:
          print("Anomaly detected, stopping workflow.")
          return END
      except Exception as e:
        raise response_error(str(e))
  
    def route_quantity(self, state:State_ax) -> Literal["ordinary_box_extraction", "cardbox_extraction"]:
      try:
        if state["primary"]["is_big_brown_cardboard_box"] == False:
          return "ordinary_box_extraction"
        else:
          return "cardbox_extraction"
      except Exception as e:
        raise response_error(str(e))

    async def crop_and_upload_node(self, state: dict, padding: int = 20) -> dict:
      """
      Node untuk preprocessing (crop pakai YOLO) lalu upload hasil ke GCS.
      Mengembalikan image_url hasil upload.
      """
      try:
          urls = []
          for image_bytes in state["images"]:
              np_arr = np.frombuffer(image_bytes, np.uint8)
              img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

              h, w = img.shape[:2]
              
              # crop with YOLO
              results = self.yolo_model.predict(img, iou=0.7, verbose=False)
              boxes = results[0].boxes.xyxy.cpu().numpy() if len(results) > 0 else []

              if len(boxes) > 0:
                  x_min = int(min(boxes[:, 0]))
                  y_min = int(min(boxes[:, 1]))
                  x_max = int(max(boxes[:, 2]))
                  y_max = int(max(boxes[:, 3]))

                  # add padding (clamp within image bounds)
                  x_min = max(0, x_min - padding)
                  y_min = max(0, y_min - padding)
                  x_max = min(w, x_max + padding)
                  y_max = min(h, y_max + padding)
                  crop = img[y_min:y_max, x_min:x_max]
              else:
                  crop = img
          
              success, buffer = cv2.imencode(".png", crop)
              if not success:
                  raise ValueError("Gagal encode hasil crop")
              cropped_bytes = buffer.tobytes()

              # upload each cropped image
              _, url_list = await upload_images_to_gcs([cropped_bytes], content_type="image/png")
              urls.extend(url_list)
          return {"url": urls}

      except Exception as e:
          print(f"Error in crop_and_upload_node: {e}")
          return {"error": str(e)}
      
    async def run_workflow(self, files, po_number) -> dict:
      response = None
      # Process files in memory
      image_bytes_list = []
      read_tasks = []

      # print("po_number:", po_number)
      
      # Create tasks to read all files in parallel
      for file in files:
        task = asyncio.create_task(file.read())
        read_tasks.append(task)
        
      # Wait for all reads to complete
      image_contents = await asyncio.gather(*read_tasks)
      image_bytes_list.extend(image_contents)

      if len(image_bytes_list) > 4:
        raise response_error("Number of maximum image is 4, please reduce the current number of image!")

      try:
        initial_state = {
                "messages": [],
                "images": image_bytes_list,
                "anomaly": None,
                "primary": None,
                "quantity": None,
                "bn_ed": None,
                "url": None,
                "po_number": po_number,
                "error": None,
            }
        
        graph = await self._init_workflow()
            
        response = await graph.ainvoke(initial_state)

        # Ambil hasil yang dibutuhkan
        anomaly_response = response.get("anomaly")
        if anomaly_response and anomaly_response.get("is_anomaly") == True:
          return response_success({
            "message": "Anomaly detected, please check the uploaded images.",
            "anomaly": anomaly_response,            
          })
        primary_response = response.get("primary")
        quantity_response = response.get("quantity")
        bned_response = response.get("bn_ed")
        url_response = response.get("url")
        po_number_response = response.get("po_number")
        error_response = response.get("error")

        # Susun hasil dalam JSON/dict
        raw_json = {
          "anomaly": anomaly_response,
          "primary": primary_response,
          "bn_ed": bned_response,
          "quantity": quantity_response,
          "url": url_response,
        }

        result_json = {"item_list": [{"item_name": raw_json['primary']['item_name'],
          "batch_number": raw_json['bn_ed']['batch_number'],
          "expired_date": raw_json['bn_ed']['expired_date'],
          "item_quantity": raw_json['quantity']['total_item'],
          "total_item_quantity": raw_json['quantity']['total_item_quantity'],
        }]}
        
        return result_json
      
      except Exception as e:
        print(f"An error occurred in : {e}")
        return response_error(str(e))
            
      finally:
        if response != None:
          for file_url in response["url"]:
              parsed_url = urlparse(file_url)

              blob_name = parsed_url.path.lstrip("/")

              blob_name = blob_name.replace("object_detection_opd/", "", 1)  

              blob = self.gcs_bucket.blob(blob_name)
              blob.delete()

objectax_controller = objectaxController()