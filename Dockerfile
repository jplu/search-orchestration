FROM python:3.8-alpine

RUN addgroup -S -g 1000 search && \
    adduser -S -u 1000 -h /home/search -s /bin/bash -g search search

RUN apk add gcc g++ linux-headers curl libffi-dev make rust cargo

RUN pip install --no-cache-dir -U nvidia-pyindex \
    && pip install --no-cache-dir -U gunicorn prometheus-fastapi-instrumentator uvicorn transformers tritonclient[all] fastapi elasticsearch==7.13.1

COPY ./gunicorn_conf.py /gunicorn_conf.py
COPY ./start.sh /start.sh
COPY ./app /app

RUN chmod +x /start.sh

WORKDIR /app/

RUN chown search /app

ENV PYTHONPATH=/app

EXPOSE 80

USER search

CMD ["/start.sh"]
