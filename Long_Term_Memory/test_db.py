import psycopg

conn = psycopg.connect(
    "postgresql://postgres:postgres@127.0.0.1:5433/postgres"
)

print("CONNECTED TO DOCKER POSTGRES ✅")
conn.close()
