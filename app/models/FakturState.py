from typing import Annotated, Optional, TypedDict

class GraphState(TypedDict):
    image_bytes: bytes
    image_url: Optional[str]
    supplier_name_extracted_from_llm: Optional[str]
    supplier_name_extracted_from_typesense: Optional[str]
    invoice_data_from_typesense: Optional[dict]
    header_data: Optional[dict]
    item_list_data: Optional[dict]
    final_data: Optional[dict]
    error: Optional[str]