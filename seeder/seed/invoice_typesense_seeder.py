import os
import json
import typesense
import time # Untuk last_updated

TYPESENSE_HOST = os.environ.get("TYPESENSE_HOST", "localhost")
TYPESENSE_PORT = int(os.environ.get("TYPESENSE_PORT", 8108))
TYPESENSE_API_KEY = os.environ.get("TYPESENSE_API_KEY", "xyz")

# Konfigurasi Typesense Client (SAMA)
client = typesense.Client({
    'nodes': [{
        'host': "typesense",  # Ini akan menjadi 'typesense' saat dijalankan via docker-compose
        'port': TYPESENSE_PORT,
        'protocol': 'http'
    }],
    'api_key': TYPESENSE_API_KEY,
    'connection_timeout_seconds': 10 # Beri waktu lebih untuk koneksi antar container
})

# Nama koleksi dan lokasi file template tunggal
TYPESENSE_COLLECTION_NAME = 'invoice_templates'
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SINGLE_TEMPLATE_FILE = os.path.join(BASE_DIR, 'assets', 'faktur_json_data', 'all_invoices_templates.json')

def create_typesense_collection(client, collection_name):
    # Skema KOLEKSI TETAP SAMA seperti sebelumnya, pastikan mencakup semua field
    collection_schema = {
        "name": collection_name,
        "fields": [
            {"name": "id", "type": "string"},
            {"name": "supplier_name", "type": "string", "facet": True, "sort": True},
            {"name": "supplier_aliases", "type": "string[]", "facet": True, "optional": True},
            {"name": "document_type", "type": "string", "facet": True, "optional": True},
            {"name": "version", "type": "string", "optional": True},
            {"name": "last_updated", "type": "int64", "optional": True},

            {"name": "fields_to_extract", "type": "string[]", "optional": True},

            # PO Number info
            {"name": "po_number_info.label", "type": "string", "index": False, "optional": True},
            {"name": "po_number_info.location_keywords", "type": "string[]", "index": True, "optional": True},
            {"name": "po_number_info.hint_text", "type": "string", "index": False, "optional": True},
            {"name": "po_number_info.rules", "type": "string[]", "index": False, "optional": True},
            {"name": "po_number_info.example_value", "type": "string", "index": False, "optional": True},

            # Invoice Number info
            {"name": "invoice_number_info.label", "type": "string", "index": False, "optional": True},
            {"name": "invoice_number_info.location_keywords", "type": "string[]", "index": True, "optional": True},
            {"name": "invoice_number_info.hint_text", "type": "string", "index": False, "optional": True},
            {"name": "invoice_number_info.rules", "type": "string[]", "index": False, "optional": True},
            {"name": "invoice_number_info.example_value", "type": "string", "index": False, "optional": True},

            # Invoice Date info
            {"name": "invoice_date_info.label", "type": "string", "index": False, "optional": True},
            {"name": "invoice_date_info.location_keywords", "type": "string[]", "index": True, "optional": True},
            {"name": "invoice_date_info.date_format_hint", "type": "string", "index": False, "optional": True},
            {"name": "invoice_date_info.hint_text", "type": "string", "index": False, "optional": True},
            {"name": "invoice_date_info.rules", "type": "string[]", "index": False, "optional": True},
            {"name": "invoice_date_info.example_value", "type": "string", "index": False, "optional": True},

            # Jika Anda memiliki field lain seperti total_amount_info, tambahkan skemanya juga
            {"name": "total_amount_info.label", "type": "string", "index": False, "optional": True},
            {"name": "total_amount_info.location_keywords", "type": "string[]", "index": True, "optional": True},
            {"name": "total_amount_info.hint_text", "type": "string", "index": False, "optional": True},
            {"name": "total_amount_info.example_value", "type": "string", "index": False, "optional": True},

            {"name": "currency_info.label", "type": "string", "index": False, "optional": True},
            {"name": "currency_info.location_keywords", "type": "string[]", "index": True, "optional": True},
            {"name": "currency_info.hint_text", "type": "string", "index": False, "optional": True},
            {"name": "currency_info.example_value", "type": "string", "index": False, "optional": True},

            {"name": "notes", "type": "string", "optional": True, "index": True}
        ],
        "default_sorting_field": "supplier_name"
    }

    try:
        client.collections[collection_name].delete()
        print(f"Old '{collection_name}' collection deleted.")
    except typesense.exceptions.ObjectNotFound:
        print(f"No existing '{collection_name}' collection found.")
    except Exception as e:
        print(f"Error deleting collection: {e}")

    try:
        client.collections.create(collection_schema)
        print(f"'{collection_name}' collection created.")
    except Exception as e:
        print(f"Error creating collection: {e}")

# ---- FUNGSI INI DIGANTI DARI load_templates_from_directory ----
def load_templates_from_single_file(file_path):
    all_templates = []
    if not os.path.exists(file_path):
        print(f"Error: Template file '{file_path}' not found.")
        return []

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            templates_data = json.load(f)
            if not isinstance(templates_data, list):
                print(f"Error: Expected a JSON array in '{file_path}', but got a different type.")
                return []
            
            for template in templates_data:
                # Tambahkan atau update field `last_updated` untuk setiap template
                template['last_updated'] = int(time.time())
                all_templates.append(template)
            print(f"Loaded {len(all_templates)} templates from {file_path}")
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from {file_path}: {e}")
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    return all_templates

def import_templates_to_typesense(client, collection_name, templates):
    if not templates:
        print("No templates to import.")
        return

    try:
        result = client.collections[collection_name].documents.import_(templates, {'action': 'upsert'})
        print(f"Imported {len(templates)} documents to Typesense.")
        # Optional: Check for errors in batch import result
        for item in result:
            if 'success' in item and not item['success']:
                print(f"Error importing document (id: {item.get('document', {}).get('id', 'N/A')}): {item.get('error', 'Unknown error')}")
    except Exception as e:
        print(f"Error during batch import to Typesense: {e}")

if __name__ == "__main__":
    create_typesense_collection(client, TYPESENSE_COLLECTION_NAME)
    # Panggil fungsi yang baru
    templates_to_import = load_templates_from_single_file(SINGLE_TEMPLATE_FILE)
    import_templates_to_typesense(client, TYPESENSE_COLLECTION_NAME, templates_to_import)
    print("\nTypesense setup and data import complete!")