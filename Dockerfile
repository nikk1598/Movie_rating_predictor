FROM python:3.8-slim-buster

COPY . /app
COPY frontend frontend
COPY config config
COPY data data
COPY models models
COPY report report
WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        cmake \
        build-essential \
        gcc

RUN pip install --upgrade pip && \
    pip install --ignore-installed -r requirements.txt

ENV PYTHONPATH=/app
EXPOSE 8501

CMD ["streamlit","run","frontend/main.py", "--server.maxUploadSize=1028"]