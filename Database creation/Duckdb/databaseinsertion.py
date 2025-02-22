import duckdb

# Connect to an in-memory DuckDB database
con = duckdb.connect(database='dining')

# Create a table
con.execute("""
CREATE TABLE users (
    user_id INT PRIMARY KEY,
    date_created DATETIME,
    username VARCHAR,
    email VARCHAR
);
"""
)


con.execute("""
            Create TABLE Restaurants
            
    RestaurantID SERIAL PRIMARY KEY, -- Unique identifier for each record
    CreatedAt Date, -- Timestamp for record creation
    username VARCHAR(50),
    PlaceID VARCHAR(50) UNIQUE NOT NULL, -- Place ID from the API
    Resturaant_Name VARCHAR(100) NOT NULL, -- Name of the restaurant
    City VARCHAR(100), -- City name for filtering
    URL TEXT, -- Website URL
    Rating FLOAT -- User Rating 
);""")
