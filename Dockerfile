FROM python:3.10.13-slim-bullseye

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY ./app .

EXPOSE 80

CMD [ "python3", "app.py"]
