import dash
from dash import dcc, html
import plotly.express as px
import pandas as pd
from sqlalchemy import create_engine
import os

# Database connection
DATABASE_URI = "postgresql+psycopg2://postgres:root@/cloudsql/bilguun3:us-central1:mypostgres"
engine = create_engine(DATABASE_URI)

# Fetch forecast data
def fetch_forecast_data():
    query = """
    SELECT date, forecasted_cases, alert_level
    FROM forecasted_alerts
    ORDER BY date;
    """
    return pd.read_sql(query, engine)

# Initialize Dash app
app = dash.Dash(__name__)

# Load forecast data
df = fetch_forecast_data()

# Layout
app.layout = html.Div([
    html.H1("Chicago COVID-19 Alert Dashboard"),
    dcc.Graph(
        id="forecast-chart",
        figure=px.line(
            df,
            x="date",
            y="forecasted_cases",
            color="alert_level",
            title="COVID-19 Forecasted Cases and Alert Levels"
        )
    ),
    html.Div([
        html.H3("Alert Level Descriptions"),
        html.Ul([
            html.Li("Low: Less than 50 cases"),
            html.Li("Medium: Between 50 and 100 cases"),
            html.Li("High: More than 100 cases")
        ])
    ])
])

if __name__ == "__main__":
    # Get the port from the environment variable
    port = int(os.getenv("PORT", "8000"))  # Default to 8000 if PORT is not set
    app.run_server(debug=True, host="0.0.0.0", port=port)
