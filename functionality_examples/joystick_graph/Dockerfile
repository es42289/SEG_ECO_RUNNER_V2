# Dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir django

# Run Django server
EXPOSE 8502
CMD ["python", "manage.py", "runserver", "0.0.0.0:8502"]
