FROM minio/minio:latest

EXPOSE 9000 9001

ENV MINIO_ROOT_USER=minioadmin
ENV MINIO_ROOT_PASSWORD=minioadmin

VOLUME ["/data"]

CMD ["server", "/data", "--console-address", ":9001"]
