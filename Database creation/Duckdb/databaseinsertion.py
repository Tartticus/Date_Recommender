import duckdb

import pandas as pd
con = duckdb.connect(database='dining')
places = pd.read_csv(r"C:/Users/Matth/Projects/Date_Recommender/places.csv")
# Prepare the INSERT statement
username = "tartticus"
insert_query = ("""INSERT INTO Restaurants 
    (RestaurantID, username, Restaurant_Type,Resturant_Name, City,Description,Rating)
VALUES (?, ?, ?, ?, ?, ?, ?)""")

for idx, row in places.iterrows():
  address = row['name'] + row['city']
  geocode_result = gmaps.geocode(address)

if geocode_result:
    # Extract the location (latitude and longitude) from the first result.
    location = geocode_result[0]['geometry']['location']
    lat = location['lat']
    lng = location['lng']
    
    
  RestaurantID = con.execute("SELECT MAX(RestaurantID) from Restaurants").fetchone()
  max_id = RestaurantID[0] if RestaurantID[0] is not None else 0  # Handle the case where the table is empty
  id = max_id + 1
  con.execute("""INSERT INTO Restaurants 
      (RestaurantID, username, Restaurant_Type,Restaurant_Name, City,Description,latitude,longitude,Rating)
  VALUES (?, ?, ?, ?, ?, ?, ?)""",
              (row['id'],username,row['type'],row['name'],row['city'],row['description'], lat = location['lat'],lat = location['lng'],row['Rating']))
  
  
