SEEDER = """🌱 Seeder Command Guide 🌱

This tool populates the database or search index with initial data.
Use it after setting up a new environment or when data sources are updated.

-----------------------------------------------------------------------
COMMANDS:
-----------------------------------------------------------------------
  invoke db:seed
    Runs the seeder for configuration in the config.
    Example: invoke db:seed sample
    
  invoke db
    Shows this help message.

-----------------------------------------------------------------------
CURRENT CONFIGURATION SEEDER THAT GONNA BE RUN:
-----------------------------------------------------------------------
{config}
"""

MIGRATION = """🔄 Migration Command Guide 🔄

This tool manages database schema changes through migrations.
Use it to evolve your database schema in a controlled way.
Make sure for every <database>Db.py file have a sync_db_url
And Table Model in app/Model are using declarative_base from 
Designated database

-----------------------------------------------------------------------
COMMANDS:
-----------------------------------------------------------------------
  invoke migrate:build --msg "description" [--branch branch_name] [--engine engine_name]
    Creates a new migration script with the given description.
    Example: invoke migrate:build --msg "add user table"

  invoke migrate:upgrade [--engine engine_name]
    Applies all pending migrations to update the database schema.
    
  invoke migrate:downgrade
    Reverts the last applied migration
    
  invoke migrate:history
    Shows the history of applied migrations.

  invoke migrate:check
    Shows the differentiation between the current database schema and project models.

  invoke migrate:current
    Shows the current migration version.

  invoke migrate:reset [--engine engine_name]
    Cleans version history and the migrations/versions directory.

  invoke migrate:fresh [--disable-downgrade]
    Resets migration history and starts a new version with optional downgrade.

  invoke migrate
    Shows this help message.

-----------------------------------------------------------------------
EXAMPLE WORKFLOW:
-----------------------------------------------------------------------
1. Create a new migration when you need schema changes:
   > invoke migrate:build --msg "add email column"

2. Review the generated migration script in migrations/versions/

3. Apply the migration to update your database:
   > invoke migrate:upgrade

4. If needed, rollback the last migration:
   > invoke migrate:downgrade

5. Check migration status:
   > invoke migrate:history

6. Compare current schema with models:
   > invoke migrate:check

7. View current version:
   > invoke migrate:current
   
or 

Auto upgrade (single script):
   > invoke migrate:fresh

Auto upgrade (single script with disable downgrade):
   > invoke migrate:fresh --disable-downgrade
"""
