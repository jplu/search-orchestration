# search-orchestration
Orchestrator for running a semantic search service that uses Triton Inference Server for the model inference and FAISS for the vector similarity. This service uses https://github.com/jplu/faiss-grpc-server as gRPC FAISS server.

## Build and Deploy

To build and deploy the Docker image run:
```
docker build -t <REPO>/search-orchestration:1.0.0 .
docker push <REPO>/search-orchestration:1.0.0
```

Replace `<REPO>` with your Docker username.

## Run

To properly run this image locally, execute these following command lines:
```
docker run --rm -d -p 9200:9200 -p 9300:9300 -e "discovery.type=single-node" -e "xpack.security.enabled=true" -e "ELASTIC_PASSWORD=changeme" docker.elastic.co/elasticsearch/elasticsearch:7.15.2
docker run --rm -d -p 8081:8081 -v <folder_containing_the_index>:/tmp faiss-server -file_path /tmp/<faiss_index_file> -port 8081
docker run --rm -d -p 8001:8001 -v<folder_containing_the_model_repository>:/model_repository nvcr.io/nvidia/tritonserver:21.09-py3 tritonserver --model-store=/model_repository
docker run --rm -d -e ES_LOGIN elastic -e ES_PASSWORD changeme -e ES_HOST localhost -e ES_PORT 9200 -e ES_IS_SECURE false -e FAISS_GRPC_HOST localhost -e TRITON_GRPC_HOST localhost -e DEVISE_JWT_SECRET_KEY <JWT_SECRET>
```

Replace:
* `<folder_containing_the_index>` by the local folder where the FAISS index file is.
* `<folder_containing_the_model_repository>` by the local folder where the `model_repository` that contains the models is.
* `<JWT_SECRET>` by the JWT you are using to encrypt the JSON payloads.

Last thing, be sure to have the data loaded in your local Elasticsearch instance.
