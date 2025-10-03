from config.typesenseDb import db
from config.setting import env

class SearchSupplierNameTool:
    
    def __init__(self):    
        self.db = db
    
    def search_supplier_name(self, supplier_name: str):
        
        search_parameters = {
            'collection': env.invoice_collection_name,
            'q': supplier_name,
            'query_by': 'supplier_name, supplier_aliases',
            'per_page': 1,
            'num_typos': 1,
            'prioritize_exact_match': True
        }
        
        print(f"Searching Typesense with parameters: {search_parameters}")
        
        multi_search_params = {"searches": [search_parameters]}
        
        search_response = self.db.multi_search(multi_search_params, return_raw=False)
        print("\n--------------SEARCH_RESPONSE-------------\n")
        print(search_response)
        return search_response
        
    
    def __call__(self, state):
        """Callable method for LangGraph node. Extracts supplier_name from state and performs search."""
        try:
            supplier_name = state.get("supplier_name_extracted_from_llm")
            if not supplier_name:
                return {"error": "No supplier_name provided in state"}
            
            print(f"Searching for supplier name: {supplier_name} in Typesense")
            
            result = self.search_supplier_name(supplier_name)
            
            top_result = result[0] if result else None
            
            print(f"Top search result: {top_result}")

            if top_result:
                state['supplier_name_extracted_from_typesense'] = top_result.get("supplier_name")
                state['invoice_data_from_typesense'] = top_result
            else:
                state['supplier_name_extracted_from_typesense'] = None
                state['invoice_data_from_typesense'] = None

            return {
                "supplier_name_extracted_from_typesense": top_result.get("supplier_name"),
                "invoice_data_from_typesense": top_result
            }
        
        except Exception as e:
            return {"error": f"Search failed: {str(e)}"}