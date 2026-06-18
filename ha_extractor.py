import pandas as pd
import psycopg2

# 1. Define connection parameters
conn_params = {
    "host": "localhost",
    "database": "homeassistant",
    "user": "homeassistant",
    "password": "your_password",
    "port": "5432",
}

# 2. Define the target entity
ENTITY_ID = "sensor.solaredge_current_power"

# 3. SQL Query
query = """
    SELECT 
        last_updated_ts AS timestamp,
        state
    FROM states
    WHERE entity_id = %s
      AND state NOT IN ('unknown', 'unavailable')
    ORDER BY last_updated_ts ASC;
"""

# 4. Connect and fetch data directly into Pandas
with psycopg2.connect(**conn_params) as conn:
    df = pd.read_sql_query(query, conn, params=(ENTITY_ID,))

# 5. Format timestamps and numbers
df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
df["state"] = pd.to_numeric(df["state"], errors="coerce")
df = df.dropna()

print(df.head())

# 4-alt. alternative query also fetching older data that HA may have put in its long term store to save space.
# query_stats = f"""
#     SELECT 
#         s.start_ts AS timestamp,
#         s.state AS value,
#         s.sum AS cumulative_sum
#     FROM statistics s
#     JOIN statistics_meta m ON s.metadata_id = m.id
#     WHERE m.statistic_id = '{ENTITY_ID}'
#     ORDER BY s.start_ts ASC;
# """
# df_stats = pd.read_sql_query(query_stats, engine)
# df_stats["timestamp"] = pd.to_datetime(df_stats["timestamp"], unit="s", utc=True)
