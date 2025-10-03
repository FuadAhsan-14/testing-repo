import re
from sqlalchemy import engine_from_config, pool
from alembic import context
from logging.config import fileConfig
import logging
from dotenv import load_dotenv
load_dotenv()

# Konfigurasi dasar
config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)
logger = logging.getLogger("alembic.env")

target_metadata = config.attributes.get('target_metadata')
db_names = config.get_main_option("databases")

def run_migrations_online() -> None:
    """Jalankan migrasi dalam mode 'online'."""
    engines = {}
    for name in re.split(r",\s*", db_names):
        engines[name] = {
            "engine": engine_from_config(
                config.get_section(name), prefix="sqlalchemy.", poolclass=pool.NullPool
            )
        }
    
    for name, rec in engines.items():
        engine = rec["engine"]
        with engine.connect() as connection:
            context.configure(
                connection=connection,
                target_metadata=target_metadata.get(name),
                upgrade_token="%s_upgrades" % name,
                downgrade_token="%s_downgrades" % name,
            )
            with context.begin_transaction():
                context.run_migrations(engine_name=name)

if not context.is_offline_mode():
    run_migrations_online()
