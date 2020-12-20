DOCKER_TAG=ongwaihong/

build:
	docker build -t reclibwh .
	docker tag reclibwh\:latest $(DOCKER_TAG)reclibwh\:latest

run_local:
	docker run --network host reclibwh\:latest

push:
	docker push $(DOCKER_TAG)reclibwh\:latest
