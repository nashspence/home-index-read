FROM pytorch/pytorch:latest

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    attr \
    ffmpeg \
    file \
    git \
    tzdata \
    imagemagick \
    libgomp1 \
    libmagic1 \
    dcraw \
    libraw-bin \
    ghostscript \
    && apt-get clean

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY packages/home_index_read .
ENTRYPOINT ["python3", "/app/main.py"]
