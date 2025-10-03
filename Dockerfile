# 
FROM python:3.12

ARG DOCKER_PORTS
ARG DOCKER_WORKER_COUNT
ENV DOCKER_PORTS=${DOCKER_PORTS}
ENV DOCKER_WORKER_COUNT=${DOCKER_WORKER_COUNT}

# 
WORKDIR /app

# 
COPY requirements.txt ./

# 
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# 
COPY ./ /app/

#
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl --fail http://localhost:${DOCKER_PORTS}/ || exit 1  

#
# CMD gunicorn -w ${DOCKER_WORKER_COUNT} -k uvicorn.workers.UvicornWorker main:app --bind 0.0.0.0:${DOCKER_PORTS}
CMD ["/bin/sh", "-c", "gunicorn -w ${DOCKER_WORKER_COUNT} -k uvicorn.workers.UvicornWorker main:app --bind 0.0.0.0:${DOCKER_PORTS}"]
