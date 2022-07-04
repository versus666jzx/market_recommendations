FROM python:3.10.5
COPY requirements.txt /
RUN pip install --no-cache-dir --no-deps -r requirements.txt
COPY src /opt/crowler
COPY config /opt/crowler/config
WORKDIR /opt/crowler

