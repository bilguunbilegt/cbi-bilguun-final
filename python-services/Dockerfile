# Base image for Python
FROM python:3.9-slim AS python-env

# Install system dependencies required for Prophet and Python libraries
RUN apt-get update && apt-get install -y \
    libpq-dev gcc g++ make && \
    pip install --no-cache-dir pandas sqlalchemy psycopg2 prophet dash plotly

# Set the working directory
WORKDIR /app

# Copy Python scripts into the image
COPY covid_forecasting.py .
COPY covid_dashboard.py .

# Expose the necessary port for the Python service
EXPOSE 8050

# Command to run the primary Python script
CMD ["python3", "covid_dashboard.py"]
