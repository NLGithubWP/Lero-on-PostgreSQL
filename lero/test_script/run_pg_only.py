import psycopg2
import os
import argparse
import json

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Run EXPLAIN ANALYZE on SQL queries.")
parser.add_argument("--dbname", type=str, required=True, help="Name of the database to connect to")
args = parser.parse_args()

# Database configuration
PORT = 5432
HOST = "localhost"
USER = "postgres"
PASSWORD = "123"
DB = args.dbname
CONNECTION_STR = f"dbname={DB} user={USER} password={PASSWORD} host={HOST} port={PORT}"
TIMEOUT = 30000000

# File with SQL queries
SQL_FILE_PATH = "q_test_0_merge.txt"  # Replace with the path to your SQL file

try:
    # Connect to PostgreSQL database
    connection = psycopg2.connect(CONNECTION_STR)
    connection.autocommit = True
    cursor = connection.cursor()

    # Read the SQL queries from the file
    with open(SQL_FILE_PATH, 'r') as file:
        lines = file.readlines()

    # Extract and execute each query from q1 to q22
    for line in lines:
        query_id, query = line.split("#####")
        query_id = query_id.strip()

        if query_id.startswith("q") and 1 <= int(query_id[1:]) <= 22:
            explain_query = f"EXPLAIN (ANALYZE, TIMING, VERBOSE, COSTS, SUMMARY, FORMAT JSON) {query}"
            cursor.execute(explain_query)
            explain_result = cursor.fetchall()[0][0]
            print(explain_result)
            # explain_data = json.loads(explain_result[0])
            # print(explain_data)
            execution_time = explain_result[0].get("Execution Time", "N/A")
            # print(f"\nExplanation for {query_id}:\n", json.dumps(explain_data, indent=2))
            print(f"Execution Time for {query_id}: {execution_time} ms")

except psycopg2.Error as e:
    print("Error: Unable to connect or execute the SQL query")
    print(e)

finally:
    if cursor:
        cursor.close()
    if connection:
        connection.close()