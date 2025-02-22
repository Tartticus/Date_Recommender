
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
)


con.execute("""
            Create TABLE Restaurants (
            
    RestaurantID INT PRIMARY KEY,
    CreatedAt Date, 
    Username VARCHAR(50),
    Resturant_Name VARCHAR(100) NOT NULL, 
    City VARCHAR(100), 
    Type TEXT,
    Description TEXT,
    UserRating FLOAT 
);""")



exclude_df['CreatedAt'] = exclude_df['CreatedAt'].apply(lambda x: datetime.strptime(x, "%m/%d/%Y").strftime("%Y-%m-%d"))


con.execute("Insert into Restaurants SELECT * FROM exclude_df")
            
            

            
            
            
            
