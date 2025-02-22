import pandas as pd

places = pd.read_csv(Date_Recommender/utility/places.csv)
# Prepare the INSERT statement
insert_query = """INSERT INTO Restaurants 
    (RestaurantID, username, Restaurant_Type,Resturant_Name, City,Description,Rating)
VALUES (?, ?, ?, ?, ?, ?, ?)"""

for idx, row in df.iterrows():
  RestaurantID = ("SELECT MAX(RestaurantID) from Restaurants").fetchone() + 1
  con.execute(insert_query,row['id'],username,row['type'],row['name'],row['city'],row['description'], row['Rating'])
