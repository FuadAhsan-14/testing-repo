# python3.12 & python3.12-venv
python3 -m venv env 
source env/bin/activate  // env\Scripts\activate

pip freeze > requirements.txt
uvicorn main:app --reload

# supervisor
## cd /etc/supervisor/conf.d
supervisord -c /etc/supervisor/supervisord.conf
supervisorctl restart app
supervisorctl status app
supervisorctl stop all

docker pull qdrant/qdrant
docker run -it --name qdrant_db_staging -p 6333:6333 -d qdrant/qdrant
