import os
import runpy
import json
import sys
from tqdm import tqdm
from typing import Literal, List, Optional
from pathlib import Path
from alembic.config import Config
from alembic import command
import shutil
import importlib

SEEDER_DIR = "seeder"
COMMAND_DIR = "app/command"
MIGRATIONS_DIR = "migrations"
CONFIG_DIR = "seeder/config.json"

def seeder_load():
    try:
        seed_dir = os.path.join(SEEDER_DIR, 'seed')
        if not os.path.isdir(seed_dir):
            print(f"❌ Error: 'seed' directory not found at {seed_dir}")
            return
        
        if not os.path.exists(CONFIG_DIR):
            print(f"\n❌ Error: Config file '{CONFIG_DIR}' not found.")
            return

        if os.path.exists(CONFIG_DIR):
            with open(CONFIG_DIR, 'r') as f:
                config_data = json.load(f)
            
            files_to_run = config_data.get('run_order', [])
            if not files_to_run:
                print("⚠️ Config file is empty or missing 'run_order' key.")
                return

            for file in tqdm(files_to_run, desc="Running Seeders from Config"):
                file_path = os.path.join(seed_dir, file)
                if os.path.exists(file_path):
                    runpy.run_path(file_path, run_name='__main__')
                else:
                    print(f"\n⚠️ Warning: Seeder '{file}' from config not found. Skipping.")
        
    except json.JSONDecodeError:
        print("❌ Error: 'seed_config.json' is not a valid JSON file.")
    except Exception as e:
        print(f"❌ An error occurred: {e}")


def select_script_from_menu(scripts: List[Path]) -> Optional[Path]:
    print("\nAvailable scripts to run:")
    for i, script_path in enumerate(scripts, 1):
        print(f"  {i}. {script_path.name}")
    print("  0. Exit")

    while True:
        try:
            choice_str = input("\nSelect a script to run (enter number): ")
            choice = int(choice_str)

            if choice == 0:
                return None
            if 1 <= choice <= len(scripts):
                return scripts[choice - 1]
            
            print(f"⚠️ Invalid selection. Please enter a number between 0 and {len(scripts)}.")

        except ValueError:
            print("❌ Invalid input. Please enter a whole number.")
        except (KeyboardInterrupt, EOFError):
            print("\nOperation cancelled by user.")
            return None

def execute_script(script_path: Path) -> None:
    print(f"\n--- Running '{script_path.name}' ---")
    try:
        runpy.run_path(str(script_path), run_name="__main__")
    except Exception as e:
        print(f"\n❌ --- Error during '{script_path.name}' execution ---", file=sys.stderr)
        print(f"   {type(e).__name__}: {e}", file=sys.stderr)
    finally:
        print(f"--- Finished '{script_path.name}' ---")


def get_dynamic_database_configs():
    """
    Memindai, memuat config DB, memuat semua model terkait, dan mengembalikan metadata/URL.
    """
    all_metadata = {}
    all_urls = {}

    project_root = os.getcwd()
    sys.path.insert(0, project_root)
    
    # --- Langkah A: Muat SEMUA model dari app/models ---
    models_dir = os.path.join(project_root, "app", "models")
    if os.path.isdir(models_dir):
        for file_name in os.listdir(models_dir):
            if file_name.endswith('.py') and not file_name.startswith('__'):
                is_likely_model = False
                full_file_path = os.path.join(models_dir, file_name)
                try:
                    with open(full_file_path, 'r', encoding='utf-8') as f:
                        file_content = f.read()
                        if '(Base)' in file_content or 'import Base' in file_content:
                            is_likely_model = True
                except Exception as e:
                    print(e)
                    is_likely_model = True 
                
                if is_likely_model:
                    module_name = file_name[:-3]
                    try:
                        importlib.import_module(f"app.models.{module_name}")
                    except ImportError as e:
                        print(f"    - Failed to load model '{module_name}': {e}")
    
    # --- Langkah B: Temukan file config DB dan kumpulkan metadata ---
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

        except (ImportError, AttributeError): 
            continue
            
    return all_metadata, all_urls

def migrations_command(
    command_type: Literal["upgrade", "downgrade", "revision", "history", "check", "current", "stamp", "reset"],
    message: str = None,
    branch: str = None,
    engine: str = None
):
    """
    Handles all Alembic database migration commands.
    A new command 'reset' is added to clear the versions folder.
    """
    # This ensures Alembic can find your project's modules
    project_root = os.getcwd()
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    # Special command to reset the versions directory
    if command_type == "reset":
        if os.path.exists(MIGRATIONS_DIR):
            try:
                for file in os.listdir(MIGRATIONS_DIR):
                    file_path = os.path.join(MIGRATIONS_DIR, file)
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
            except Exception as e:
                print(f"❌ Error while deleting versions: {e}")
        return

    # Standard Alembic command processing
    alembic_cfg = Config("core/alembic.ini")
    alembic_cfg.set_main_option('prepend_sys_path', project_root)

    try:
        all_metadata, all_urls = get_dynamic_database_configs()
    except Exception as e:
        print(f"❌ Failed to get database configs: {e}")
        return
    
    if engine:
        if engine not in all_urls:
            print(f"❌ Error: Engine '{engine}' not found. Options: {list(all_urls.keys())}")
            return
        all_urls = {engine: all_urls[engine]}
        all_metadata = {engine: all_metadata.get(engine)}
        print(f"🎯 Targeting only engine: {engine}")

    if not all_metadata:
        print("No database configurations found.")
        return

    alembic_cfg.attributes['target_metadata'] = all_metadata
    for engine_name, url in all_urls.items():
        if not alembic_cfg.file_config.has_section(engine_name):
            alembic_cfg.file_config.add_section(engine_name)
        alembic_cfg.set_section_option(engine_name, "sqlalchemy.url", url)

    db_names = ",".join(all_urls.keys())
    alembic_cfg.set_main_option("databases", db_names)

    # Use a dictionary for cleaner command mapping
    commands = {
        "upgrade": lambda: command.upgrade(alembic_cfg, "heads"),
        "downgrade": lambda: command.downgrade(alembic_cfg, "-1"),
        "revision": lambda: command.revision(alembic_cfg, message=message, autogenerate=True, branch_label=branch),
        "history": lambda: command.history(alembic_cfg),
        "check": lambda: command.check(alembic_cfg),
        "current": lambda: command.heads(alembic_cfg),
        "stamp": lambda: command.stamp(alembic_cfg, "base", purge=True)
    }

    if command_type in commands:
        try:
            commands[command_type]()
        except Exception as e:
            print(f"❌ An error occurred during '{command_type}': {e}") if command_type != "check" else print(str(e))
    else:
        print(f"Command '{command_type}' not recognized.")
