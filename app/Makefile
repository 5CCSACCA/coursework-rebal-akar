# Variables
KIND_CLUSTER_NAME = hatespeech-cluster
AUTH_IMAGE_NAME = auth-service:latest
PREDICTION_IMAGE_NAME = prediction-service:latest
PROMETHEUS_CONFIG = kubernetes/prometheus.yaml
K8S_NAMESPACE = hatespeech
PROM_NAMESPACE = hatespeech   # Changed from monitoring to hatespeech
DOCKER_NETWORK = kind
DOCKER_CONTEXT = .
KUBECTL = kubectl
KIND = kind
NGINX_NAMESPACE = ingress-nginx

# Targets
.PHONY: all build kind-cluster load-images nginx ingress apply prometheus clean delete-cluster

all: build kind-cluster load-images nginx apply prometheus ingress

# Build the Docker images
build:
	docker build -t $(AUTH_IMAGE_NAME) ./auth_service
	docker build -t $(PREDICTION_IMAGE_NAME) ./prediction_service

# Create the Kind cluster
kind-cluster:
	$(KIND) create cluster --name $(KIND_CLUSTER_NAME) --config kind-config.yaml || echo "Cluster already exists"

# Load Docker images into Kind
load-images:
	$(KIND) load docker-image $(AUTH_IMAGE_NAME) --name $(KIND_CLUSTER_NAME)
	$(KIND) load docker-image $(PREDICTION_IMAGE_NAME) --name $(KIND_CLUSTER_NAME)

# Deploy Nginx ingress controller
nginx:
	$(KUBECTL) apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/main/deploy/static/provider/kind/deploy.yaml
	@echo "Waiting for Nginx ingress controller to be ready..."
	$(KUBECTL) wait --namespace $(NGINX_NAMESPACE) \
	  --for=condition=ready pod \
	  --selector=app.kubernetes.io/component=controller \
	  --timeout=90s

# Apply the Kubernetes configurations for namespace, secrets, and services
apply:
	$(KUBECTL) apply -f kubernetes/namespace.yaml
	$(KUBECTL) apply -f kubernetes/secrets.yaml -n $(K8S_NAMESPACE)
	$(KUBECTL) apply -f kubernetes/auth_service/ -n $(K8S_NAMESPACE)
	$(KUBECTL) apply -f kubernetes/prediction_service/ -n $(K8S_NAMESPACE)

# Apply the Prometheus configuration
prometheus:
	$(KUBECTL) apply -f $(PROMETHEUS_CONFIG)
	@echo "Waiting for Prometheus to be ready..."
	$(KUBECTL) wait --namespace $(PROM_NAMESPACE) \
	  --for=condition=ready pod \
	  --selector=app=prometheus \
	  --timeout=90s

# Apply the ingress resource
ingress:
	$(KUBECTL) apply -f kubernetes/ingress.yaml -n $(K8S_NAMESPACE)

# Clean up all resources
clean:
	$(KUBECTL) delete namespace $(K8S_NAMESPACE) || echo "Namespace does not exist"
	$(KUBECTL) delete -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/main/deploy/static/provider/kind/deploy.yaml || echo "Nginx ingress already deleted"

# Delete the Kind cluster
delete-cluster:
	$(KIND) delete cluster --name $(KIND_CLUSTER_NAME)
