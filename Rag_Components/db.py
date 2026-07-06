import os
from dotenv import load_dotenv
from psycopg_pool import ConnectionPool
from psycopg import OperationalError

#================
# Database
#================
# psycopg_pool
# psycopg[binary]



load_dotenv()

# Create one global pool
pool = ConnectionPool(
    conninfo=os.getenv("DATABASE_URL"),
    min_size=2,
    max_size=20,
)

try:
    # Test the database connection
    with pool.connection() as conn:
        print("✅ Database connected successfully!")

        with conn.cursor() as cur:

            # Create table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS students (
                    id SERIAL PRIMARY KEY,
                    name TEXT NOT NULL,
                    age INT NOT NULL
                )
            """)

            # Insert data
            cur.execute(
                "INSERT INTO students (name, age) VALUES (%s, %s)",
                ("Chelwa", 18)
            )

            # Read data
            cur.execute("SELECT * FROM students")

            rows = cur.fetchall()

            print("\nStudents:")
            for row in rows:
                print(row)

            # Save changes
            conn.commit()

except OperationalError as e:
    print("❌ Database connection failed!")
    print(e)

except Exception as e:
    print("❌ Error:", e)

finally:
    # Close the pool before exiting
    pool.close()
    print("✅ Connection pool closed.")