import logging
import os
import sys

from typing import List

import grpc
import numpy as np
import tritonclient.grpc as grpcclient

from elasticsearch import Elasticsearch
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.logger import logger as fastapi_logger
from jose import JWTError, jwt
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel, HttpUrl
from starlette.responses import Response, JSONResponse
from starlette.status import HTTP_200_OK
from transformers import DistilBertTokenizer

import faiss_pb2_grpc
import faiss_pb2


class DocumentItem(BaseModel):
    url: HttpUrl
    context: str
    pid: int
    title: str
    id: int


class Token(BaseModel):
    access_token: str
    token_type: str


class OrchestrationException(Exception):
    def __init__(self, query: str, message: str, from_svc: str):
        self.query = query
        self.message = message
        self.from_svc = from_svc


es_login = os.getenv("ES_LOGIN")
es_password = os.getenv("ES_PASSWORD")
es_host = os.getenv("ES_HOST")
grpc_faiss_host = os.getenv("FAISS_GRPC_HOST")
grpc_triton_host = os.getenv("TRITON_GRPC_HOST")
secret_key = os.getenv("DEVISE_JWT_SECRET_KEY")

es_client = Elasticsearch(
    [es_host],
    http_auth=(es_login, es_password),
    scheme="https",
    port=443,
    timeout=300,
    retry_on_timeout=True,
    max_retries=10,
    maxsize=25,
)

tokenizer = DistilBertTokenizer.from_pretrained('sentence-transformers/msmarco-distilbert-base-v3', use_fast=True)

try:
    faiss_channel = grpc.insecure_channel(f'{grpc_faiss_host}:8081')
    stub = faiss_pb2_grpc.FaissServiceStub(faiss_channel)
except Exception as e:
    print("FAISS channel creation failed: " + str(e))
    sys.exit()

model_name = "encode_onnx"

try:
  triton_client = grpcclient.InferenceServerClient(
    url=f"{grpc_triton_host}:8001"
  )
except Exception as e:
    print("Triton channel creation failed: " + str(e))
    sys.exit()

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

app = FastAPI()

Instrumentator().instrument(app).expose(app)

gunicorn_error_logger = logging.getLogger("gunicorn.error")
gunicorn_logger = logging.getLogger("gunicorn")
uvicorn_access_logger = logging.getLogger("uvicorn.access")

uvicorn_access_logger.handlers = gunicorn_error_logger.handlers
fastapi_logger.setLevel(gunicorn_logger.level)


def encode_with_triton(query):
    inputs = []
    outputs = []
    model_input = tokenizer.encode_plus(query, truncation=True)
    input_ids = np.expand_dims(model_input["input_ids"], axis=0)
    attention_mask = np.expand_dims(model_input["attention_mask"], axis=0)
    
    inputs.append(grpcclient.InferInput('input_ids', [1, len(model_input["input_ids"])], "INT64"))
    inputs.append(grpcclient.InferInput('attention_mask', [1, len(model_input["attention_mask"])], "INT64"))

    inputs[0].set_data_from_numpy(input_ids)
    inputs[1].set_data_from_numpy(attention_mask)

    outputs.append(grpcclient.InferRequestedOutput('sentence_embedding'))

    results = triton_client.infer(
        model_name=model_name,
        inputs=inputs,
        outputs=outputs,
        headers={'test': '1'}
    )

    return results.as_numpy('sentence_embedding')[0]


def nearest_documents_faiss(encoded_query, k):
    faiss_request = faiss_pb2.SearchRequest()
    faiss_request.top_k = k
    faiss_request.vector.float_val.extend(encoded_query)
    response = stub.Search.future(faiss_request, 90)
    result = response.result()
    ids = []

    for res in result.neighbors:
        ids.append(res.id)

    return ids


def get_es_documents(ids):
    documents = []

    for id in ids:
        doc = es_client.get(index="en_wikipedia", id=id)
        documents.append(doc['_source'])

    return documents

@app.exception_handler(OrchestrationException)
async def orchestration_exception_handler(request, exc):
    fastapi_logger.error(f"""
        ======================
        Query
        {str(exc.query)}
        Exception
        {str(exc.message)}
        From
        {str(exc.from_svc)}
        ======================
    """)
    
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )

@app.post('/search', response_model=List[DocumentItem])
async def search(token: str = Depends(oauth2_scheme), k: int = 5):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, secret_key, algorithms=["HS256"])
    except JWTError:
        raise credentials_exception

    query = payload.get("query")
    
    try:
        encoded_query = encode_with_triton(query)
    except Exception as e:
        raise OrchestrationException(query=query, message=str(e), from_svc="Triton")
    
    try:
        nearest_document_ids = nearest_documents_faiss(encoded_query, k)
    except Exception as e:
        raise OrchestrationException(query=query, message=str(e), from_svc="FAISS")
    
    try:
        documents = get_es_documents(nearest_document_ids)
    except Exception as e:
        raise OrchestrationException(query=query, message=str(e), from_svc="Elasticsearch")
    
    return documents


@app.get('/health')
async def health():
    return Response(status_code=HTTP_200_OK)

@app.get('/ready')
async def ready():
    return Response(status_code=HTTP_200_OK)

@app.get('/')
async def root():
    return Response(status_code=HTTP_200_OK)
