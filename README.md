# search-orchestration
Orchestrator for running a semantic search service that uses Triton Inference Server for the model inference and FAISS for the vector similarity. This service uses https://github.com/jplu/faiss-grpc-server as gRPC FAISS server.

## Build and Deploy

To build and deploy the Docker image run:
```
docker build -t <REPO>/search-orchestration:1.0.0 .
docker push <REPO>/search-orchestration:1.0.0
```

Replace `<REPO>` with your Docker username.
