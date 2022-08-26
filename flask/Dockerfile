FROM python:3.8-slim

# Copy everything to destination .
COPY . . 

#COPY serve.py /app

#COPY ./distilbert-based-uncased-argumentativewriting /app

#WORKDIR /app

RUN pip install -r requirements.txt

CMD python serve.py
