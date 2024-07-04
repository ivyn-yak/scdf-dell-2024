FROM --platform=linux/amd64 bitnami/tensorflow-serving:2.15.1

WORKDIR /app

USER root 

RUN apt-get update \
    && apt-get install -y python3-full python3-venv python3-pip pkg-config libhdf5-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .

RUN /opt/venv/bin/pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8080

USER 1001

CMD ["/opt/venv/bin/uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]