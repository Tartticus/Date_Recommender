from main.py import pick_location, preprocess_text, Train_Model, Fetch_Restaurants, Reccomend_Restaurants, insert_place_chosen
import os
import datetime
import duckdb
import googlemaps
import pandas as pd

from flask import Flask, render_template, request, redirect, url_for, session

# Initialize Flask app
app = Flask(__name__)
app.secret_key = "Lizard69"

# Set up DuckDB connections
# In‑memory DB for processing data
conn = duckdb.connect(database=':memory:', read_only=False)
conn.execute("""
CREATE TABLE IF NOT EXISTS processing_restaurants (
    name TEXT,
    vicinity TEXT,
    reviews TEXT
);
""")
conn.execute("""
CREATE TABLE IF NOT EXISTS processed_restaurants (
    name TEXT,
    vicinity TEXT,
    reviews TEXT,
    predicted_score TEXT
);
""")
# Permanent DB connection (adjust your path as needed)
con = duckdb.connect(database=r'C:/Users/Matth/Projects/Date_Recommender/dining')

# Initialize Google Maps client
api_key = os.getenv("Google_Places_API_Key")
gmaps_client = googlemaps.Client(key=api_key)


# -------------------------
# Flask Routes
# -------------------------
@app.route('/', methods=['GET', 'POST'])
def home():
    options = ['Dining', 'Dessert', 'Japanese', 'Ramen', 'Taiwanese', 'Bar', 'Italian']
    if request.method == 'POST':
        # Get form inputs
        session['username'] = request.form.get('username')
        session['restaurant_type'] = request.form.get('restaurant_type')
        session['city'] = request.form.get('city')
        return redirect(url_for('process'))
    return render_template('home.html', options=options)

@app.route('/process')
def process():
    # Retrieve session variables
    username = session.get('username')
    restaurant_type = session.get('restaurant_type')
    city = session.get('city')
    
    # Get geocode data for the city
    geocode_result = gmaps_client.geocode(city)
    if geocode_result:
         lat = geocode_result[0]['geometry']['location']['lat']
         lng = geocode_result[0]['geometry']['location']['lng']
         state = None
         for component in geocode_result[0]['address_components']:
             if 'administrative_area_level_1' in component['types']:
                 state = component['long_name']
                 break
         session['lat'] = lat
         session['lng'] = lng
         session['state'] = state
    else:
         return "Error retrieving location data.", 500

    # Query for training data from your existing Restaurants table
    query = f"""SELECT Description, 
                CASE WHEN Rating >= 7 THEN 'High' 
                     WHEN Rating >= 4 THEN 'Medium' 
                     ELSE 'Low' END AS score 
                FROM Restaurants 
                WHERE Restaurant_Type = '{restaurant_type}'"""
    df = con.execute(query).df()
    # (You might wish to add error-handling if df is empty)
    
    # Train the model
    naive_bayes, vectorizer = Train_Model(df)
    
    # Fetch additional restaurants from Google Maps
    Fetch_Restaurants(city, session.get('state'))
    
    # Get recommendations based on the fetched reviews
    names, restaurant_list = Reccomend_Restaurants(naive_bayes, vectorizer)
    session['restaurant_list'] = restaurant_list
    # Pass recommendations to template (as a list of dicts)
    return render_template('recommend.html', restaurants=names.to_dict(orient='records'))


@app.route('/map')
def show_map():
    import folium
    import duckdb

    # Connect to your DuckDB
    con = duckdb.connect(database=r'C:/Users/Matth/Projects/Date_Recommender/dining')

    # Retrieve restaurant data
    df = con.execute("SELECT Restaurant_Name, Restaurant_Type, latitude, longitude, Rating FROM Restaurants").df()

    # Center the map on the mean location of restaurants
    center_lat = df['latitude'].mean()
    center_lon = df['longitude'].mean()
    m = folium.Map(location=[center_lat, center_lon], zoom_start=13)

    # Define colors for each restaurant type
    colors = [
        'blue', 'green', 'red', 'purple', 'orange', 'darkred', 'lightred',
        'beige', 'darkblue', 'darkgreen', 'cadetblue', 'darkpurple', 'white',
        'pink', 'lightblue', 'lightgreen', 'gray', 'black', 'lightgray'
    ]
    restaurant_types = df['Restaurant_Type'].unique()
    color_dict = {rtype: colors[i % len(colors)] for i, rtype in enumerate(restaurant_types)}

    # Add markers for each restaurant
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

    # Get HTML representation of the map
    map_html = m._repr_html_()
    return render_template("map.html", map_html=map_html)


@app.route('/select', methods=['POST'])
def select():
    # Get the chosen restaurant (the radio button value is the index)
    selected_index = int(request.form.get('restaurant_index'))
    restaurant_list = session.get('restaurant_list')
    selected_restaurant = restaurant_list[selected_index]
    session['selected_restaurant'] = selected_restaurant
    
    # Get place details and insert into your permanent DB
    place_id, url = insert_place_chosen(selected_restaurant, session.get('city'))
    current_datetime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    username = session.get('username')
    restaurant_type = session.get('restaurant_type')
    lat = session.get('lat')
    lng = session.get('lng')
    state = session.get('state')
    
    RestaurantID = con.execute("SELECT MAX(RestaurantID) from Restaurants").fetchone()
    max_id = RestaurantID[0] if RestaurantID[0] is not None else 0
    RestaurantID = max_id + 1
    con.execute("""INSERT INTO Restaurants 
        (RestaurantID, CreatedAt, username, PlaceID, Restaurant_Type, Resturant_Name, City, State, latitude, longitude, URL)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (RestaurantID, current_datetime, username, place_id, restaurant_type, 
         selected_restaurant, session.get('city'), state, lat, lng, url))
    
    return render_template('confirmation.html', restaurant=selected_restaurant, url=url)

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
