FROM tensorflow/tensorflow:latest as base

RUN  apt-get -y update && apt install -y git libpq-dev

FROM base

ADD . /opt/recommender/
WORKDIR /opt/recommender
RUN  pip install .
RUN chmod +x /opt/recommender/run.sh
WORKDIR /opt/recommender

CMD ["/bin/sh", "/opt/recommender/run.sh"]