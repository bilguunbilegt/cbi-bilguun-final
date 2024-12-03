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
    df = pd.read_sql(query, engine)
    return df

# Preprocess data
def preprocess_data(df):
    # Ensure the data is sorted and properly formatted
    df['ds'] = pd.to_datetime(df['ds'])
    return df

# Forecasting
def forecast_covid_alerts(df):
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

# Save forecast to database
def save_forecast_to_db(forecast):
    forecast.rename(columns={'ds': 'date', 'yhat': 'forecasted_cases'}, inplace=True)
    forecast.to_sql('forecasted_alerts', engine, if_exists='replace', index=False)

if __name__ == "__main__":
    data = fetch_covid_data()
    preprocessed_data = preprocess_data(data)
    forecast = forecast_covid_alerts(preprocessed_data)
    save_forecast_to_db(forecast)
    print("Forecasting completed and saved to database.")
