FROM python:3.8-slim
WORKDIR /srv
RUN pip install --no-cache-dir pipenv
ENV PIPENV_SYSTEM=true
COPY . .
RUN pipenv install
