docker run -d -p 8080:8080 -v $(pwd)/data/metadata.db:/data/metadata.db sql-web
docker run -p 6333:6333   -v $(pwd)/qdrant_storage:/qdrant/storage    qdrant/qdrant
