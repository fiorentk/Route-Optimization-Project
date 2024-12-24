
FROM python:3.10.8

ENV PYTHONDONTWRITEBYTECODE 1

WORKDIR /code
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt
COPY ./app /code/.
