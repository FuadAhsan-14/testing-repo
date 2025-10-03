import os
import sys
import importlib

# def load_all_models_and_base_metadata(models_dir: str = "app/models"):
#     """
#     Dynamically loads all SQLAlchemy models and returns the Base class.
#     """
#     engine = load_db_module_filter_base()

#     module_path_prefix = models_dir.replace('/', '.') + '.'
#     model_files = [f for f in os.listdir(models_dir) if f.endswith('.py') and not f.startswith('__')]
#     for file_name in model_files:
#         module_name = os.path.splitext(file_name)[0]
#         full_module_path = module_path_prefix + module_name
#         try:
#             importlib.import_module(full_module_path)
#         except ImportError as e:
#             print(f"❌ Error loading model {full_module_path}: {e}")
        
#     # Return the dynamically loaded Base class
#     return engine

# def load_db_module_filter_base(disable_filter:bool = False):
#     config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config')
#     db_modules = [f[:-3] for f in os.listdir(config_path) if f.endswith('.py') and 'Db' in f]
#     if disable_filter:
#         return db_modules
#     base = {}
#     for env_engine in db_modules:
#         base_module_path = f"config.{env_engine}"
#         try:
#             base_module = importlib.import_module(base_module_path)
#             Base = getattr(base_module, 'Base')
#             base[env_engine] = Base.metadata
#         except (ImportError, AttributeError) as e:
#             print(f"❌ Could not import 'Base' from module '{base_module_path}': {e}")
    
#     return base

def get_dynamic_database_configs():
    """
    Memindai, memuat config DB, memuat semua model terkait, dan mengembalikan metadata/URL.
    """
    all_metadata = {}
    all_urls = {}

    project_root = os.getcwd()
    sys.path.insert(0, project_root)
    
    models_dir = os.path.join(project_root, "app", "models")
    if os.path.isdir(models_dir):
        for file_name in os.listdir(models_dir):
            if file_name.endswith('.py') and not file_name.startswith('__'):
                module_name = file_name[:-3]
                try:
                    importlib.import_module(f"app.models.{module_name}")
                except ImportError as e:
                    print(f"    - Gagal mengimpor model '{module_name}': {e}")
    
    config_path = os.path.join(project_root, 'config')
    if not os.path.isdir(config_path):
        raise FileNotFoundError(f"Direktori config tidak ditemukan di: {config_path}")

    db_files = [f for f in os.listdir(config_path) if f.endswith('Db.py') and not f.startswith('__')]

    for file_name in db_files:
        module_name = file_name[:-3]
        dict_key = module_name.replace('Db', '').lower()
        
        module_path = f"config.{module_name}"
        try:
            module = importlib.import_module(module_path)
            Base = getattr(module, 'Base')
            sync_url = getattr(module, 'SYNC_DB_URL')

            if Base.metadata.tables:
                all_metadata[dict_key] = Base.metadata
                all_urls[dict_key] = sync_url

        except (ImportError, AttributeError) as e: 
            continue
            
    return all_metadata, all_urls

if __name__=="__main__":
    print(get_dynamic_database_configs())
