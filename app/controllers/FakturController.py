import asyncio
import os
import uuid
import io
from PIL import Image
from fastapi import HTTPException
from app.generative import manager
from langgraph.graph import StateGraph, START, END
from app.models.FakturState import GraphState
from app.services.faktur.SupplierNameAgentService import SupplierNameAgentService
from app.services.faktur.GeneralHeaderAgentService import GeneralHeaderAgentService
from app.services.faktur.TypesenseHeaderAgentService import TypesenseHeaderAgentService
from app.services.faktur.ItemListAgentService import ItemListAgentService
from app.tools.SearchSupplierNameTool import SearchSupplierNameTool
from app.utils.Http.HttpResponseUtils import response_error, response_success
from config.setting import env
from config.credentials import google_credential
from app.utils.Uploader.GcpUploaderUtils import GcpUloader

class FakturAutoEvalController:
    def __init__(self):
        
        llm_supplier_name = manager._get_llm(name="gemini_model_flash_lite25", temperature=0)
        llm_item_list = manager._get_llm(name="gemini_model_flash_lite25", temperature=0)
        llm_header = manager._get_llm(name="gemini_model_flash_lite25", temperature=0)
        
        self.uploader = GcpUloader(
            project_name=env.app_name,
            bucket_name=env.bucket_name,
            credentials=google_credential()
        )
        
        self.SupplierNameAgent = SupplierNameAgentService(llm=llm_supplier_name)
        self.GeneralHeaderAgent = GeneralHeaderAgentService(llm=llm_header)
        self.TypesenseHeaderAgent = TypesenseHeaderAgentService(llm=llm_header)
        self.ItemListAgent = ItemListAgentService(llm=llm_item_list)
        self.SearchSupplierNameTool = SearchSupplierNameTool()

    async def _init_workflow(self):
        
        graph = StateGraph(GraphState)

        graph.add_node("upload_image", self.upload_image_node)
        
        graph.add_node("extract_supplier_name", self.SupplierNameAgent)
        graph.add_node("search_typesense", self.SearchSupplierNameTool)
        graph.add_node("extract_header_with_template", self.TypesenseHeaderAgent)
        graph.add_node("extract_header_generic", self.GeneralHeaderAgent)
        graph.add_node("header_finished", self.header_finished_node)
        graph.add_node("extract_item_list", self.ItemListAgent)
        graph.add_node("format_final_response", self.format_final_response_node)

        graph.add_edge(START, "upload_image")
        graph.add_edge("upload_image", "extract_supplier_name")
        graph.add_edge("upload_image", "extract_item_list")

        graph.add_conditional_edges(
            "extract_supplier_name", self.route_header_extraction,
            {"use_typesense": "search_typesense", "use_generic": "extract_header_generic"}
        )
        graph.add_conditional_edges(
            "search_typesense", lambda s: "use_template" if s.get("invoice_data_from_typesense") else "use_generic",
            {"use_template": "extract_header_with_template", "use_generic": "extract_header_generic"}
        )

        graph.add_edge("extract_header_with_template", "header_finished")
        graph.add_edge("extract_header_generic", "header_finished")
        graph.add_edge(["header_finished", "extract_item_list"], "format_final_response")
        graph.add_edge("format_final_response", END)

        workflow = graph.compile()
        
        workflow.get_graph().draw_mermaid_png(output_file_path=f"app\\controllers\\visualization\\{self.__class__.__name__}.png")
        
        return workflow

    def route_header_extraction(self, state: GraphState) -> str:
        """Memutuskan jalur ekstraksi header berdasarkan nama supplier."""
        if state.get("error"):
            return "use_generic"
            
        supplier_name = state.get("supplier_name_extracted_from_llm")
        
        if supplier_name and supplier_name.strip():
            return "use_typesense"
        else:
            return "use_generic"
        
    def header_finished_node(self, state: GraphState) -> dict:
        """
        Node dummy yang berfungsi sebagai titik keluar terpadu untuk
        semua kemungkinan jalur ekstraksi header.
        """
        print("Header extraction branch has finished.")
        return {}

    async def upload_image_node(self, state: GraphState) -> dict:
        """Node untuk mengunggah gambar ke GCS menggunakan GcpUloader."""
        try:
            image_bytes = state['image_bytes']

            unique_id = uuid.uuid4().hex
            file_upload_name = f"faktur_{unique_id}.png"
            
            public_url = await self.uploader.upload_bytes(
                file_bytes=image_bytes,
                blob_path=file_upload_name,
                content_type="image/png",
                make_public=True
            )

            print(f"Image uploaded to: {public_url}")
            return {"image_url": public_url}
        except Exception as e:
            print(f"Error in upload_image_node: {e}")
            return {"error": str(e)}
    
    async def _cleanup_gcs_image(self, url: str):
        """Membersihkan gambar dari GCS setelah selesai diproses."""
        if not url: return
        try:
            from urllib.parse import urlparse
            parsed_url = urlparse(url)
            blob_name = os.path.basename(parsed_url.path)
            
            # Dapatkan bucket dari uploader
            bucket = self.uploader._bucket
            blob = bucket.blob(blob_name)
            
            # Jalankan delete di executor
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, blob.delete)

            print(f"Cleaned up image: {blob_name}")
        except Exception as e:
            print(f"Failed to cleanup GCS image {url}: {e}")
            
    def format_final_response_node(self, state: GraphState) -> dict:
        """
        Node untuk menggabungkan data, menghitung confidence score, dan membangun
        output JSON dengan urutan kunci yang rapi dan terstruktur.
        """
        if state.get("error"):
            return {"error": state.get("error")}

        header_data = state.get('header_data', {})
        item_list_from_state = state.get('item_list_data', {})
        item_list_data = item_list_from_state.get('item_list', []) if isinstance(item_list_from_state, dict) else []

        supplier_name = state.get('supplier_name_extracted_from_typesense') or \
                        state.get('supplier_name_extracted_from_llm')
                        
        print(f"Formatting final response with supplier_name: {supplier_name}")
        print(f"Header data: {header_data}")
        print(f"Item list data: {item_list_data}")

        final_item_list = []
        
        for item in item_list_data:
            ordered_item = {
                'item_name': item.get('item_name'),
                'quantity': item.get('quantity'),
                'uom': item.get('uom'),
                'batch_number': item.get('batch_number'),
                'expired_date': item.get('expired_date')
            }
            final_item_list.append(ordered_item)

        final_data = {
            "supplier_name": supplier_name,
            "po_number": header_data.get("po_number"),
            "invoice_number": header_data.get("invoice_number"),
            "invoice_date": header_data.get("invoice_date"),
            "item_list": final_item_list
        }
        
        return {"final_data": final_data}

    async def run_workflow(self, files: list) -> dict:

        if not files or len(files) > 1:
            raise HTTPException(status_code=400, detail="Please upload exactly one image.")
        
        file = files[0]
        image_url = None

        try:
            image_bytes = await file.read()
            
            initial_state = {
                "messages": [],
                "image_bytes": image_bytes,
                "image_url": None,
                "supplier_name_extracted_from_llm": None,
                "supplier_name_extracted_from_typesense": None,
                "invoice_data_from_typesense": None,
                "header_data": None,
                "item_list_data": None,
                "final_data": None,
                "error": None,
            }
            
            graph = await self._init_workflow()
            
            response = await graph.ainvoke(initial_state)
            
            image_url = response.get("image_url")

            if response.get("error"):
                raise Exception(response["error"])

            return response_success(response.get("final_data", {}))

        except Exception as e:
            print(f"An error occurred in faktur_scan: ")
            # return response_error(str(e))
            
        finally:
            if image_url:
                asyncio.create_task(self._cleanup_gcs_image(image_url))
        
fakturautoeval_controller = FakturAutoEvalController()