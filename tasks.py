import json
import ast
from invoke import task, Collection, Task
from pathlib import Path

from core.entrypoint import seeder_load, execute_script, migrations_command
from core.static import SEEDER, MIGRATION

def confirm_action(prompt: str) -> bool:
    """Helper function to get user confirmation for destructive actions."""
    confirmation = input(f"{prompt} [y/N]: ")
    return True if confirmation.lower() in ['y', 'yes'] else False

@task
def seeder_seed(c):
    """Loads seed data into the database using seed configuration"""
    seeder_load()
    
@task
def seeder_info(c):
    """Displays usage examples for all seeder commands."""
    with open('seeder/config.json', 'r') as f:
        config = json.load(f)
        text = '\n'.join([f"{i}. {seed}" for i, seed in enumerate(config['run_order'], 1)])
        print(SEEDER.format(config=text))        
  
@task
def migration_build(c, msg: str = None, branch=None, engine=None):
    """Builds a new migration script."""
    migrations_command("revision", msg, branch, engine)

@task
def migration_upgrade(c, engine=None):
    """Executes database migrations."""
    migrations_command("upgrade", engine=engine)
    
@task 
def migration_downgrade(c):
    """Execute revert database migration"""
    confirmation = confirm_action("Are you sure you want to downgrade the database? This action may cause data loss in the database")
    if confirmation:
        migrations_command("downgrade")

@task
def migration_history(c):
    """Shows the revision history."""
    migrations_command("history")

@task
def migration_check(c):
    """Shows the differentiation between the current database schema and project models"""
    migrations_command("check")

@task
def migration_current(c):
    """Shows the current migration version"""
    migrations_command("current")

@task
def migration_auto(c, disable_downgrade:bool = False):
    """Resets migration history and starts a new version with optional downgrade."""    
    confirmation = confirm_action("Are you sure you want to reset the migration history and start new version? \nAny column data that are deleted cannot be restored. This cannot be undone.")
    if confirmation:
        migrations_command("reset")
        migrations_command("stamp")
        migrations_command("revision", "init")
        migrations_command("upgrade")
        if disable_downgrade:
            migrations_command("stamp")
            migrations_command("reset")

@task 
def migration_reset(c):
    """Cleans version history and the migrations/versions directory by removing all version."""
    confirmation = confirm_action("Are you sure you want to reset the migration history? This cannot be undone.")
    if confirmation:
        migrations_command("reset")
        migrations_command("stamp") 

@task 
def migration_info(c):
    """Displays usage examples for all migration commands."""
    print(MIGRATION)

@task 
def run_test(c):
    """Runs pytest to execute unit tests, system tests and integration test."""
    c.run("pytest --color=yes")

def load_all_commands(collection):
    """
    Dynamically finds, parses, and loads all command scripts into a collection.
    """
    command_dir = Path("app/command")
    for script_path in command_dir.glob("*.py"):
        try:
            source = script_path.read_text(encoding="utf-8")
            task_func = lambda ctx, path=script_path: execute_script(script_path=path)
            task_func.__doc__ = ast.get_docstring(ast.parse(source)) or ""
            collection.add_task(Task(task_func), name=f"command:{script_path.stem}")

        except (IOError, SyntaxError) as e:
            print(f"⚠️  Skipping invalid command file '{script_path.name}': {e}")


# collection
ns = Collection()
ns.add_task(seeder_info, 'db')
ns.add_task(seeder_seed, 'db:seed')

ns.add_task(migration_build, 'migrate:build')
ns.add_task(migration_upgrade, 'migrate:upgrade')
ns.add_task(migration_downgrade, 'migrate:downgrade')
ns.add_task(migration_history, 'migrate:history')
ns.add_task(migration_check, 'migrate:check')
ns.add_task(migration_current, 'migrate:current')
ns.add_task(migration_reset, 'migrate:reset')
ns.add_task(migration_auto, 'migrate:fresh')
ns.add_task(migration_info, 'migrate')
ns.add_task(run_test, 'test')

# ns.add_task(command, 'command')
load_all_commands(ns)


