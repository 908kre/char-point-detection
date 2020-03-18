FROM python:3.8-slim
RUN apt-get update \
    && apt-get install -y --no-install-recommends inotify-tools gcc libc-dev unzip
ENV MYPYPATH=/srv/stubs
WORKDIR /srv
COPY . .
RUN pip install -e .[dev]
