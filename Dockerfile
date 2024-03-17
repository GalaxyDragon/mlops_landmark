FROM python:3.9
LABEL authors="aberdovskiy"
ARG BUILD_DEPS="curl"
RUN apt-get update && apt-get install -y $BUILD_DEPS
WORKDIR ./
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
COPY . .

RUN pip3 install -r requirements.txt