
import duckdb
from datetime import datetime
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



con.execute("""
            Create TABLE Restaurants(
            
    RestaurantID INTEGER PRIMARY KEY, 
    PlaceID VARCHAR(50) UNIQUE NOT NULL, -- Place ID from the API
    CreatedAt Date, -- Timestamp for record creation
    Resturant_Name VARCHAR(50) NOT NULL, -- Name of the restaurant
    username VARCHAR(50),
    
    
    Description NVARCHAR(300),
    City VARCHAR(50), -- City name for filtering
    State VARCHAR(50),
    latitude DOUBLE,
    longitude DOUBLE,
    URL TEXT, -- Website URL
    Restaurant_Type VARCHAR (50),
    Rating FLOAT -- User Rating 
);""")




            
            

            
            
            
            
