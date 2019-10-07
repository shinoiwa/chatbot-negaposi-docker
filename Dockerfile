FROM kyobad/miniconda3-alpine:latest

MAINTAINER K.Kato

RUN pip install --upgrade pip \
    && conda install -y flask \
    && pip install flask gunicorn line-bot-sdk janome scikit-learn\
    && adduser -D botter \
    && mkdir /home/botter/app

USER botter

COPY ./app /home/botter/app

WORKDIR /home/botter/app

CMD gunicorn -b 0.0.0.0:$PORT bot:app --log-file=-
