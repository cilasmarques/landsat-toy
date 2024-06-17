# Install container toolkit 
> https://medium.com/@avishkumar27/how-to-monitor-nvidia-gpus-using-dcgm-prometheus-and-grafana-430836ae47bd

> https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html

* Install
```sh
sudo apt-get update
sudo apt-get install -y nvidia-docker2
```

* Run
```sh
docker network create monitoring
docker run -d --gpus all --rm --network monitoring --name dcgm-exporter -p 9400:9400 nvcr.io/nvidia/k8s/dcgm-exporter:3.3.0-3.2.0-ubuntu22.04
docker run -d   --name prometheus --network monitoring  -p 9090:9090   -v ./prometheus.yml:/etc/prometheus/prometheus.yml   prom/prometheus
docker run -d   --name=grafana   --network monitoring   -p 3000:3000   grafana/grafana
```