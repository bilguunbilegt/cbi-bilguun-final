# Chicago Business Intelligence for Strategic Planning

This project uses microservices to collect and process data from the City of Chicago Data Portal for strategic planning and analysis.

## Microservices Overview

### 1. **Taxi Trips Data Collection**
   - Collects taxi trip data from the City of Chicago's datasets.
   - URL: `https://data.cityofchicago.org/resource/wrvz-psew.json`

### 2. **Community Area Unemployment Data Collection**
   - Collects unemployment and related public health statistics.
   - URL: `https://data.cityofchicago.org/resource/iqnk-2tcu.json`

### 3. **Building Permits Data Collection**
   - Collects building permits data from the city portal.
   - URL: `https://data.cityofchicago.org/resource/ydr8-5enu.json`

### 4. **COVID-19 Data Collection**
   - Collects weekly COVID-19 cases, rates, and test data.
   - URL: `https://data.cityofchicago.org/resource/yhhz-zm2v.json`

### 5. **CCVI Data Collection**
   - Collects Community Vulnerability Index (CCVI) data.
   - URL: `https://data.cityofchicago.org/resource/xhc6-88s9.json`

---

## Installation and Deployment Steps

### Prerequisites
- Google Cloud account.
- Installed: [gcloud CLI](https://cloud.google.com/sdk/docs/install), [Docker](https://www.docker.com/), and Go.

### Steps to Deploy

1. **Setup Cloud SQL Database**
   - Create a PostgreSQL instance in Google Cloud SQL with the name `mypostgres`.
   - Create a database `chicago_business_intelligence`.

2. **Deploy pgAdmin**
   - Use the provided `cloudbuild.yaml` file:
     ```yaml
     - name: "gcr.io/cloud-builders/docker"
       args: ['pull', 'dpage/pgadmin4']
     ```
   - Deploy pgAdmin to Cloud Run to manage your database.

3. **Build and Deploy Go Microservice**
   - Navigate to the project directory.
   - Build and push the Docker image:
     ```bash
     docker build -t gcr.io/<PROJECT-ID>/go-microservice .
     docker push gcr.io/<PROJECT-ID>/go-microservice
     ```
   - Deploy the microservice to Cloud Run using:
     ```bash
     gcloud run deploy go-microservice \
       --image gcr.io/<PROJECT-ID>/go-microservice \
       --region us-central1 \
       --add-cloudsql-instances <PROJECT-ID>:us-central1:mypostgres \
       --platform managed \
       --port 8080 \
       --allow-unauthenticated
     ```

4. **Environment Variables**
   - Set the following environment variables in Cloud Run for the microservice:
     - `DB_USER=postgres`
     - `DB_PASSWORD=<your-db-password>`
     - `DB_NAME=chicago_business_intelligence`
     - `INSTANCE_CONNECTION_NAME=<PROJECT-ID>:us-central1:mypostgres`

---

## Running Locally
1. Clone the repository and navigate to the project folder.
2. Modify the database connection string in `main.go`:
   ```go
   db_connection := "user=postgres dbname=chicago_business_intelligence password=<your-password> host=localhost sslmode=disable port=5432"
   ```
3. Start the service:
   ```bash
   go run main.go
   ```
4. Access the service at `http://localhost:8080`.

---

## Notes
- Use pgAdmin for database management.
- Review and update the `cloudbuild.yaml` and `Dockerfile` if deploying with different configurations.

For issues or questions, contact your administrator or refer to the Google Cloud [troubleshooting guide](https://cloud.google.com/docs/troubleshooting).


