
## Project Overview

HateSpeech-Offensive SaaS is a scalable Software-as-a-Service (SaaS) application designed to detect hate speech and offensive language in tweets. Utilizing machine learning, specifically a fine-tuned DistilBERT model, the service provides RESTful APIs for user authentication and batch prediction of tweet sentiments. The application is containerized and deployed using Kubernetes, ensuring high availability and scalability.

## Features

- **User Authentication:** Secure user registration and login with JWT-based authentication.
- **Batch Predictions:** Analyze up to 100 tweets per request to classify them as Neutral, Offensive, or Hate.
- **Prediction History:** Retrieve past predictions with pagination support.
- **Scalable Deployment:** Containerized microservices deployed on Kubernetes for high availability and scalability.
- **Monitoring:** Integrated Prometheus metrics for monitoring service health and performance.
- **Comprehensive Logging:** Structured JSON logging for easy debugging and monitoring.

## Architecture

The application consists of two main microservices:

1. **Authentication Service (`auth_service`):** Handles user registration, login, and JWT token generation.
2. **Prediction Service (`prediction_service`):** Processes tweet predictions and manages prediction history.

## Technologies Used

- **Backend Framework:** FastAPI
- **Authentication:** JWT (JSON Web Tokens) via `python-jose`
- **Password Hashing:** `passlib` with bcrypt
- **Database:** MongoDB Atlas
- **Machine Learning:** PyTorch, Transformers (DistilBERT)
- **Containerization:** Docker
- **Orchestration:** Kubernetes
- **Monitoring:** Prometheus
- **Testing:** Pytest, Pytest-Asyncio
- **Version Control:** GitFlow

## Prerequisites

- **Docker:** Ensure Docker is installed and running.
- **kubectl**
- **Kind**
- **Python 3.9**
- **Git LFS:** Ensure Git LFS is downloaded and configured
    sudo apt update
    sudo apt install -y git-lfs
    *Initialize git lfs in environment*
    git lfs install

  ```bash
  sudo apt install git-lfs
  ```

## Deployment Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/5CCSACCA/coursework-rebal-akar
cd coursework
cd app
```

### 2. Run MakeFile for Kubernetes Deployment

```bash
make all
```

If make is not present in your system download it.

To check status of pods:

```bash
kubectl get pods -n hatespeech
```

### API Documentation

Access authentication service via: [http://localhost/auth/docs](http://localhost/auth/docs)  
Access prediction service via: [http://localhost/predict/docs](http://localhost/predict/docs)  
Access Prometheus UI via: [http://localhost/]

#### Authentication Service:

**Register User**

- **Endpoint:** `POST /auth/users/register`
- **Description:** Register a new user with unique username and email
- **Request Body:**

  ```json
  {
    "username": "johndoe",
    "email": "johndoe@example.com",
    "password": "Password123!"
  }
  ```

- **Response:**

  ```json
  {
    "id": "60d0fe4f5311236168a109ca",
    "username": "johndoe",
    "email": "johndoe@example.com",
    "created_at": "2023-12-03T12:34:56.789Z",
    "updated_at": "2023-12-03T12:34:56.789Z"
  }
  ```

**Login User**

- **Endpoint:** `POST /auth/users/login`
- **Description:** Authenticate a user and receive a JWT token. Password must include digit, special character, uppercase and lowercase letter. Must be 8 characters long. In the input 
- **Request Body:**

Ensure " " are NOT added. For example if you registered with "test", in login username simply add test. 

- **Response:**

  ```json
  {
    "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
    "token_type": "bearer"
  }
  ```

#### Prediction Service:

**Make Prediction**

- **Endpoint:** `POST /predict/predict`
- **Description:** Analyze a batch of tweets
- **Headers:** `Authorization: Bearer <access_token>`
  - In `/docs` UI, simply click on the authorize button and enter user credentials in prediction service
- **Request Body**

  ```json
  {
    "input_texts": [
      "I hate this!",
      "This is great."
    ]
  }
  ```

- **Response:**

  ```json
  {
    "predictions": [
      {
        "prediction": 2,
        "prediction_label": "Hate",
        "probabilities": [0.05, 0.10, 0.85]
      },
      {
        "prediction": 0,
        "prediction_label": "Neutral",
        "probabilities": [0.90, 0.05, 0.05]
      }
    ],
    "hate_offensive_tweets": [
      {
        "prediction": 2,
        "prediction_label": "Hate",
        "probabilities": [0.05, 0.10, 0.85]
      }
    ]
  }
  ```

**Retrieve Predictions**

- **Endpoint:** `GET /predict/predictions`
- **Description:** Fetch a list of past predictions
- **Query Parameters:**
  - `skip` (optional): Number of records to skip
  - `limit` (optional): Number of records to retrieve

- **Response:**

  ```json
  [
    {
      "id": "60d0fe4f5311236168a109cb",
      "user_id": "60d0fe4f5311236168a109ca",
      "input_text": "I hate this!",
      "prediction_result": {
        "prediction": 2,
        "prediction_label": "Hate",
        "probabilities": [0.05, 0.10, 0.85]
      },
      "created_at": "2023-12-03T12:45:00.123Z"
    },
    ...
  ]
  ```

### Running Tests

If you would like to run unit tests:

For `auth_service`:

```bash
cd auth_service
pip install -r requirements.txt
pytest
```

For `prediction_service`:

```bash
cd prediction_service
pip install -r requirements.txt
pytest
```

### Monitoring and Logging

**Prometheus:** At [http://localhost/], you can access various parameters on your pods

**Logging:** To access logs

```bash
kubectl get pods -n hatespeech
```

```bash
kubectl logs <pod-name> -n hatespeech
```

To view logs from all containers:

```bash
kubectl logs --all-containers=true -n hatespeech
```
