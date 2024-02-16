FROM python:3.10.13-slim-bullseye

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY ./app .

EXPOSE 8050

CMD [ "python3", "plot_eraser.py"]
