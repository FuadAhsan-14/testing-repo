import asyncio
import json
from typing import Literal
from urllib.parse import urlparse
import cv2
import numpy as np
from starlette.exceptions import HTTPException
from langgraph.graph import START, StateGraph, END
from google.cloud import storage
from app.utils.Helper import upload_images_to_gcs
from config.setting import env
from config.credentials import google_credential
from app.generative.manager import manager
from app.models.ObjectState import State_ax
from app.services.objectaxBackup.PrimaryAgent import PrimaryAgentService
from app.services.objectaxBackup.AnomalyAgent import AnomalyAgentService
from app.services.objectaxBackup.OrdinaryAgent import OrdinaryBoxAgentService
from app.services.objectaxBackup.CardBoxAgent import CardboxAgentService
from app.services.objectaxBackup.BnEdAgent import BnedAgentService
from app.utils.Http.HttpResponseUtils import response_success, response_error 

class objectaxBackupController:
    def __init__(self):

        llm_anomaly = manager._get_llm(name="claude_model_sonnet_35", temperature=0)
        llm_primary = manager._get_llm(name="claude_model_sonnet_35", temperature=0)
        llm_cardbox = manager._get_llm(name="claude_model_sonnet_37", temperature=0)
        llm_ordinary = manager._get_llm(name="claude_model_sonnet_37", temperature=0)
        llm_bned = manager._get_llm(name="openai_model_gpt_mini_5", temperature=0)

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

        builder.add_edge(START, "anomaly_detection") 
        builder.add_conditional_edges("anomaly_detection", self.route_anomaly)
        builder.add_conditional_edges("primary_extraction", self.route_quantity)
        builder.add_edge("primary_extraction", "bned_extraction")
        builder.add_edge("bned_extraction", END)
        builder.add_edge("cardbox_extraction", END)
        builder.add_edge("ordinary_box_extraction", END) 

        compiled_graph = builder.compile()

        compiled_graph.get_graph().draw_mermaid_png(output_file_path=f"app\\controllers\\visualization\\{self.__class__.__name__}.png")

        return compiled_graph

    async def route_anomaly(self, state:State_ax) -> Literal["primary_extraction", END]:
      try:
        if state["anomaly"]["is_anomaly"] == False:
          return "primary_extraction"
        else:
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
        quantity_response = response.get("quantity")
        bned_response = response.get("bn_ed")
        anomaly_response = response.get("anomaly")
        primary_response = response.get("primary")
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

objectaxBackup_controller = objectaxBackupController()