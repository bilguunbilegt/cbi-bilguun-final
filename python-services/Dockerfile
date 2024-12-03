# Base image for Python
FROM python:3.9-slim AS python-env

# Install system dependencies required for Prophet and Python libraries
RUN apt-get update && apt-get install -y \
    libpq-dev gcc g++ make curl && \
    pip install --no-cache-dir pandas sqlalchemy psycopg2 prophet dash plotly pystan==2.19.1.1

# Install the Cloud SQL Auth Proxy
RUN curl -o /cloud_sql_proxy https://dl.google.com/cloudsql/cloud_sql_proxy.linux.amd64 && \
    chmod +x /cloud_sql_proxy

# Set the working directory
WORKDIR /app

# Copy Python scripts into the image
COPY covid_forecasting.py .
COPY covid_dashboard.py .

# Copy the service account key for authentication
COPY service-account-key.json /app/service-account-key.json
ENV GOOGLE_APPLICATION_CREDENTIALS="/app/service-account-key.json"

# Expose the port for Cloud Run
EXPOSE 8000

# Set the PORT environment variable (default to 8000)
ENV PORT 8000

# Run the Cloud SQL Auth Proxy and the Python scripts
CMD ["bash", "-c", "/cloud_sql_proxy -instances=bilguun3:us-central1:mypostgres=tcp:5432 -credential_file=/app/service-account-key.json & python3 covid_forecasting.py && python3 covid_dashboard.py"]
