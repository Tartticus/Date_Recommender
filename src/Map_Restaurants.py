import folium
import duckdb
con = duckdb.connect(database=r'C:/Users/Matth/Projects/Date_Recommender/dining')

df = con.execute("SELECT Restaurant_Name, Restaurant_Type, latitude, longitude, Rating FROM Restaurants").df()

# Center the map based on the mean location
center_lat = df['latitude'].mean()
center_lon = df['longitude'].mean()
m = folium.Map(location=[center_lat, center_lon], zoom_start=13)

# Map each restaurant type to a color.
# Define a list of colors. Adjust as needed.
colors = [
    'blue', 'green', 'red', 'purple', 'orange', 'darkred', 'lightred',
    'beige', 'darkblue', 'darkgreen', 'cadetblue', 'darkpurple', 'white',
    'pink', 'lightblue', 'lightgreen', 'gray', 'black', 'lightgray'
]
# Create a dictionary mapping each unique Restaurant_Type to a color.
restaurant_types = df['Restaurant_Type'].unique()
color_dict = {rtype: colors[i % len(colors)] for i, rtype in enumerate(restaurant_types)}

# Add markers to the map.
for _, row in df.iterrows():
    folium.CircleMarker(
        location=[row['latitude'], row['longitude']],
        radius=6,
        popup=f"{row['Restaurant_Name']}<br>Rating: {row['Rating']}",
        color=color_dict[row['Restaurant_Type']],
        fill=True,
        fill_color=color_dict[row['Restaurant_Type']],
        fill_opacity=0.7
    ).add_to(m)

# Save the map to an HTML file
m.save("restaurants_map.html")
print("Map saved as restaurants_map.html")
