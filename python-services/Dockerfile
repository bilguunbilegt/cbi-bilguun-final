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

# Expose the port for Cloud Run
EXPOSE 8050

# Set the PORT environment variable (default to 8050)
ENV PORT 8050

# Run the forecasting script before starting the dashboard
CMD ["bash", "-c", "python3 covid_forecasting.py && python3 covid_dashboard.py"]
