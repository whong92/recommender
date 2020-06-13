FROM tensorflow/tensorflow:latest-py3


ADD . /opt/recommender/
WORKDIR /opt/recommender
RUN  apt-get -y update && \
    apt install -y git libpq-dev && \
    pip install .
RUN chmod +x /opt/recommender/run.sh
WORKDIR /opt/recommender

CMD ["/bin/sh", "/opt/recommender/run.sh"]