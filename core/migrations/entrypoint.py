import sys
import os
import shutil
from alembic.config import Config
from alembic import command
from core.migrations.retrieve_base import get_dynamic_database_configs
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
print(project_root)
sys.path.insert(0, project_root)


def run_alembic(type: str, message: str = None, branch: str = None, engine=None):
    """Membangun konfigurasi dan menjalankan perintah Alembic."""
    
    alembic_cfg = Config("core/alembic.ini")
    project_root = os.getcwd()
    alembic_cfg.set_main_option('prepend_sys_path', project_root)
    all_metadata, all_urls = get_dynamic_database_configs()
    
    if engine:
        if engine not in all_urls:
            print(f"❌ Error: Engine '{engine}' tidak ditemukan. Pilihan: {list(all_urls.keys())}")
            return
        all_urls = {engine: all_urls[engine]}
        all_metadata = {engine: all_metadata.get(engine)}
        print(f"🎯 Menargetkan hanya pada engine: {engine}")

    if not all_metadata:
        print("Tidak ada konfigurasi database yang ditemukan.")
        return

    alembic_cfg.attributes['target_metadata'] = all_metadata
    for engine_name, url in all_urls.items():
        if not alembic_cfg.file_config.has_section(engine_name):
            alembic_cfg.file_config.add_section(engine_name)
        alembic_cfg.set_section_option(engine_name, "sqlalchemy.url", url)

    db_names = ",".join(all_urls.keys())
    alembic_cfg.set_main_option("databases", db_names)

    match type:
        case "upgrade":
            command.upgrade(alembic_cfg, "heads")
        
        case "downgrade":
            command.downgrade(alembic_cfg, "-1")

        case "revision":
            command.revision(alembic_cfg, message=message, autogenerate=True, branch_label=branch)

        case "history":
            command.history(alembic_cfg)
    
        case "check":
            command.check(alembic_cfg)
    
        case "current":
            command.heads(alembic_cfg)
    
        case "stamp":
            command.stamp(alembic_cfg, "base", purge=True)
    
        case _:
            print(f"Perintah '{type}' tidak dikenali.")
            pass

def delete_version(versions_dir: str = "migrations/versions"):
    if not os.path.exists(versions_dir):
        print(f"Directory {versions_dir} does not exist")
        return False
    try:
        for file in os.listdir(versions_dir):
            file_path = os.path.join(versions_dir, file)
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        return True
    except Exception as e:
        print(f'Error: {e}')
        return False
