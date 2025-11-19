# Mindsight Machine Learning Demo # 
A demo repository illustrating a full-stack machine-learning application:

## Key components
backend API (in api/)
machine-learning model/training code (in ml/)
frontend application (in frontend/)
containerised via docker-compose.yml

## Before you begin, make sure you have these installed:

1. Python

Version 3.9 or later recommended
Needed if you want to run or develop the ML code locally (outside Docker)

Verify installation:
python --version

2. Docker

Docker Engine
Docker Compose (usually included with Docker Desktop)
Required to run the full application stack

Verify installation:

docker --version
docker compose version

##  Getting Started

1. Clone the repository

git clone https://github.com/vikideak/mindsight-machine-learning-demo.git
cd mindsight-machine-learning-demo

2. Run the project

For local running / model training:
pip install -r requirements.txt

Run the entire project using Docker Compose
docker compose up --build

To stop everything:
docker compose down

## Project Structure

```
/ (root)
├── api/             # Backend API service
├── ml/              # Machine learning training code & model files
├── frontend/        # Frontend UI
├── requirements.txt # Python dependencies
├── docker-compose.yml
└── .gitignore
```

## Usage

After running docker compose up, the frontend will be available at http://localhost:8080
The frontend talks to the backend API service (port defined in docker-compose).

For retraining the model run python ml/train.py
Containers need to be rebuilt after retraining the model:
docker compose build
docker compose up
