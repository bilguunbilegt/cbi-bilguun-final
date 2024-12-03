import pandas as pd
from prophet import Prophet
from sqlalchemy import create_engine

# Database connection (replace placeholders with actual values)
DATABASE_URI = "postgresql+psycopg2://postgres:root@/cloudsql/bilguun3:us-central1:mypostgres"

engine = create_engine(DATABASE_URI)

# Fetch data from the database
def fetch_covid_data():
    query = """
    SELECT week_start AS ds, cases_weekly AS y
    FROM covid_details
    WHERE cases_weekly IS NOT NULL
    ORDER BY week_start;
    """
    try:
        df = pd.read_sql(query, engine)
        if df.empty:
            raise ValueError("No data found in 'covid_details' table.")
        return df
    except Exception as e:
        print(f"Error fetching data: {e}")
        raise

# Preprocess data
def preprocess_data(df):
    try:
        # Convert dates and handle invalid entries
        df['ds'] = pd.to_datetime(df['ds'], errors='coerce')  # Invalid dates become NaT
        invalid_dates = df[df['ds'].isna()]  # Log invalid dates
        if not invalid_dates.empty:
            print("Invalid dates found and removed:")
            print(invalid_dates)
        
        df = df.dropna(subset=['ds'])  # Drop rows with invalid dates
        return df
    except Exception as e:
        print(f"Error during preprocessing: {e}")
        raise

# Forecasting
def forecast_covid_alerts(df):
    try:
        if df.empty or len(df) < 2:
            raise ValueError("Insufficient data to train the Prophet model.")
        
        model = Prophet()
        model.fit(df)

        future = model.make_future_dataframe(periods=14, freq='D')  # Forecast next 14 days
        forecast = model.predict(future)

        # Assign alert levels based on forecasted cases
        forecast['alert_level'] = pd.cut(
            forecast['yhat'],
            bins=[-float('inf'), 50, 100, float('inf')],
            labels=['Low', 'Medium', 'High']
        )
        return forecast[['ds', 'yhat', 'alert_level']]
    except Exception as e:
        print(f"Error during forecasting: {e}")
        raise

# Save forecast to database
def save_forecast_to_db(forecast):
    try:
        if forecast.empty:
            raise ValueError("No forecast data to save.")
        
        forecast.rename(columns={'ds': 'date', 'yhat': 'forecasted_cases'}, inplace=True)
        forecast.to_sql('forecasted_alerts', engine, if_exists='replace', index=False)
        print("Forecast saved successfully to database.")
    except Exception as e:
        print(f"Error saving forecast to database: {e}")
        raise

if __name__ == "__main__":
    try:
        print("Fetching COVID-19 data from database...")
        data = fetch_covid_data()
        print(f"Data fetched: {data.head()}")

        print("Preprocessing data...")
        preprocessed_data = preprocess_data(data)
        print(f"Preprocessed data: {preprocessed_data.head()}")

        print("Running forecast...")
        forecast = forecast_covid_alerts(preprocessed_data)
        print(f"Forecast: {forecast.head()}")

        print("Saving forecast to database...")
        save_forecast_to_db(forecast)
        print("Forecasting completed successfully.")
    except Exception as main_exception:
        print(f"An error occurred: {main_exception}")
