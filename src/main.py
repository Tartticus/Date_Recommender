import json
import numpy as np
import requests
import nltk
import googlemaps
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
from nltk.stem import WordNetLemmatizer, PorterStemmer
import duckdb
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from datetime import datetime
import os
import gmaps
import datetime
nltk.download('punkt_tab')
nltk.download('stopwords')
print("imports complete\n")



#%% Restaurant selection and upload


def pick_location():
    username = input("What is your username?:\n")
    # Prompt the user to pick a restaurant from the DataFrame
    
    options = ['Dining','Dessert','Japanese','Ramen','Taiwanese','Bar','Italian']
        # Print available options
    print("Please choose one of the following options:\n")
    for i, option in enumerate(options, 1):
        print(f"{i}. {option}")
    while True:   
        # Prompt the user for input
        choice = input("Enter the number corresponding to your choice:\n ")
    
        # Validate the user's input
        if not choice.isdigit() or int(choice) < 1 or int(choice) > len(options):
            print("Invalid choice. Please try again.")
        
        else:
            selected_option = options[int(choice) - 1]
            print(f"You selected: {selected_option}\n")    
            break
      
    gmaps = googlemaps.Client(key=api_key)
    
    #user input for cities
    city = input("Enter which city you want to visit:\n")
    
    # Request autocomplete suggestions restricted to cities
    selected_city = gmaps.places_autocomplete(
        input_text=city,
        types="(cities)"
    )
    
    
    
        
    try:
                # Convert the selected city to coordinates
                geocode_result = gmaps_client.geocode(selected_city)
                if geocode_result:
                    lat = geocode_result[0]['geometry']['location']['lat']
                    lng = geocode_result[0]['geometry']['location']['lng']
                    print(f"Coordinates: {lat}, {lng}")
                    #Get state name
                    address_components = geocode_result[0]['address_components']
                    state = None
                    for component in address_components:
                        if 'administrative_area_level_1' in component['types']:
                            state = component['long_name']
                            
                else:
                    print("Failed to retrieve coordinates for the selected city.")
               
    except Exception as e:
            print(e)
    #Get df values out    
    df = con.execute(f"SELECT Description,  CASE WHEN Rating >= 7 THEN 'High' WHEN Rating >= 4 THEN 'Medium' ELSE 'Low' END AS score FROM Restaurants WHERE Restaurant_Type = '{selected_option}'").df()
    # Map score values to 1, 2, 3
    score_mapping = {'High': 1, 'Medium': 2, 'Low': 3}
    df['score'] = df['score'].map(score_mapping)
    print(df)
    
    return df, selected_city, state, lat, lng, username, selected_option

    

#%% NLP
# Text preprocessing function
def preprocess_text(text):
    if isinstance(text, str):
        # Text cleaning
        # Remove unwanted characters, numbers, punctuation, and symbols
        cleaned_text = text

        # Tokenization
        tokens = word_tokenize(cleaned_text)

        # Lowercasing
        tokens = [token.lower() for token in tokens]

        # Stop word removal
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words]

        # Stemming
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(token) for token in tokens]

        # Join the processed tokens back into a single string
        processed_text = ' '.join(tokens)

        return processed_text
    else:
        return ""


def Train_Model(df):
    #make score the index
    df['score'] = df['score'].astype(int)
    df['scoreINDEX'] = df['score']
    df = df.set_index('scoreINDEX')
    #convert to string
    df_string = df.to_string(index=False)
    print(df_string) 
    # Apply preprocessing to the 'description' column
    text_processed = df['Description'].apply(preprocess_text)
    
    print(df)
    
    from sklearn.metrics import classification_report
    #%% Machine learning
    
    # Create the CountVectorizer
    vectorizer = CountVectorizer()
    
    # Encode the preprocessed text data
    X = vectorizer.fit_transform(text_processed)
    y = df['score']
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the Naive Bayes classifier
    naive_bayes = MultinomialNB()
    naive_bayes.fit(X_train, y_train)
    
    # Make predictions on the testing data
    y_pred = naive_bayes.predict(X_test)
    
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    classification_report = classification_report(y_test, y_pred)
    
    print(f"Accuracy: {accuracy:.2f}")
    print("Classification Report:")
    print(classification_report)
    
    return naive_bayes, vectorizer




    
    

#%%city

def Fetch_Restaurants(selected_city,selected_state):
    exclude_df = con.execute(f"SELECT Restaurant_Name from Restaurants").df()
    # Extract the restaurant names to exclude
    excluded_restaurants = exclude_df['Restaurant_Name'].tolist()
    
    
    
    # Perform the place search in the selected city and state
    places_result = gmaps_client.places(query=f'{selected_city}, {selected_state}')
    geocode_result = gmaps_client.geocode(selected_city)
    if geocode_result:
        lat = geocode_result[0]['geometry']['location']['lat']
        lng = geocode_result[0]['geometry']['location']['lng']
    
    # Search for restaurants in the selected city and state
    restaurants_result = gmaps_client.places_nearby(type='restaurant', location= f"{lat},{lng}" , radius = 11519, open_now = True)
    
    # Search for bars in the selected city and state
    bars_result = gmaps_client.places_nearby(type='bar', location= f"{lat},{lng}" , radius = 11519, open_now = True)
    
    # Extract restaurant information
    restaurants = restaurants_result['results']
    restaurant_data = []
    for restaurant in restaurants:
        place_id = restaurant['place_id']
        place_details = gmaps_client.place(place_id=place_id, fields=['name', 'vicinity', 'reviews'])
        place_details = place_details['result']
        
        # Exclude restaurant if its name is in the excluded list
        if place_details['name'] in excluded_restaurants:
            continue
        
        reviews = place_details.get('reviews', [])
        review_data = []
        for review in reviews:
            review_data.append(review['text'])
        
        restaurant_data.append({
            'name': place_details['name'],
            'vicinity': place_details['vicinity'],
            'reviews': " ".join(review_data)
        })
    
    # Extract bar information
    bars = bars_result['results']
    bar_data = []
    for bar in bars:
        place_id = bar['place_id']
        place_details = gmaps_client.place(place_id=place_id, fields=['name', 'vicinity', 'reviews'])
        place_details = place_details['result']
        
        # Exclude bar if its name is in the excluded list
        if place_details['name'] in excluded_restaurants:
            continue
        
        reviews = place_details.get('reviews', [])
        review_data = []
        for review in reviews:
            review_data.append(review['text'])
        
        bar_data.append({
            'name': place_details['name'],
            'vicinity': place_details['vicinity'],
            'reviews': " ".join(review_data)
        })
    
      
    datadf = pd.DataFrame(restaurant_data)
    datadf2 = pd.DataFrame(bar_data)
    conn.execute("INSERT INTO processing_restaurants SELECT * FROM datadf")
    conn.execute("INSERT INTO processing_restaurants SELECT * FROM datadf2")
    print(f"Data saved to Database")


#%% Json processing (first checks for high score - then medium)
def Reccomend_Restaurants(naive_bayes,vectorizer):
    model = naive_bayes
    
    restaurants_data = conn.execute("SELECT * FROM processing_restaurants").df()
    # Convert the index to integer (if it's not already)
    restaurants_data.index = restaurants_data.index.astype(int)
    # Preprocess the restaurant reviews
    preprocessed_reviews = []
    for idx, row in restaurants_data.iterrows():
        reviews = row['reviews']
        preprocessed_reviews.extend([preprocess_text(reviews)])
    
    # Vectorize the preprocessed reviews
    X_test = vectorizer.transform(preprocessed_reviews)
    
    # Make predictions using the trained model
    y_pred = model.predict(X_test)
    
    # Map the predicted scores to labels
    score_labels = {1: 'High', 2: 'Medium', 3: 'Low'}
    predicted_scores = [score_labels[pred] for pred in y_pred]
    
    # Update the restaurant data with predicted scores
   
    restaurants_data['predicted_score'] = predicted_scores
    
    
    datadf = restaurants_data
    conn.execute("INSERT INTO processed_restaurants SELECT * FROM datadf")
     
    print(f"Data saved to DuckDB")
    
    
    
    
    names = restaurants_data[['name','predicted_score']]
        
    
    
    
    # Print the DataFrame
    print(names)
    
    while True:
        # Prompt the user for input
        choice = input("Enter the number corresponding to your choice: ")

        try:
            # Validate the user's input
            choice_index = int(choice) - 1

            if choice_index < 0 or choice_index >= len(names['name']):
                print("Invalid choice. Please try again.")
            else:
                selected_restaurant = names['name'][choice_index]
                print(f"You selected: {selected_restaurant}")
                break
        except ValueError:
            print("Invalid choice. Please enter a number.")

    return selected_restaurant

  #%% API 
def insert_place_chosen(selected_restaurant,city):
    query = f"{selected_restaurant}, {city}"
    
    find_response = gmaps_client.find_place(
        input=query,
        input_type="textquery",
        fields=['place_id', 'name']
    )
    candidates = find_response.get('candidates', [])
    if not candidates:
        print(f"No results found for: {query}")
        

    # Take the first candidate from the results.
    candidate = candidates
    place_id = candidate.get('place_id')
    
    # Now retrieve detailed information using the Place Details API.
    details_response = gmaps_client.place(
        place_id=place_id,
        fields=['name', 'formatted_address', 'reviews', 'url']
    )

    details = details_response.get('result', {})
    url = details.get("url")
    return place_id, url



#In memory db
conn = duckdb.connect(database=':memory:', read_only=False)
conn.execute("""
CREATE TABLE IF NOT EXISTS processing_restaurants (
    name TEXT,
    vicinity TEXT,
    reviews TEXT
);
"""
)

conn.execute("""
CREATE TABLE IF NOT EXISTS processed_restaurants (
    name TEXT,
    vicinity TEXT,
    reviews TEXT,
    predicted_score TEXT
);
"""
)
con = duckdb.connect(database=r'C:/Users/Matth/Projects/Date_Recommender/dining')

# Replace 'YOUR_API_KEY' with your actual API key
api_key = os.getenv("Google_Places_API_Key")
#create gmaps client
gmaps_client = googlemaps.Client(key=api_key)
def main():
    df, selected_city, state, lat, lng,username, Restaurant_Type = pick_location()
    naive_bayes, vectorizer = Train_Model(df)
    Fetch_Restaurants(selected_city,state)
    selected_restaurant= Reccomend_Restaurants(naive_bayes,vectorizer)
    try:
        place_id, url = insert_place_chosen(selected_restaurant,selected_city)
    except:
        pass
    # Get the current date and time
    current_datetime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    RestaurantID = con.execute("SELECT MAX(RestaurantID) from Restaurants").fetchone()
    max_id = RestaurantID[0] if RestaurantID[0] is not None else 0  # Handle the case where the table is empty
    RestaurantID = max_id + 1
    # Prepare the INSERT statement
    con.execute("""INSERT INTO Restaurants 
        (RestaurantID, CreatedAt, username, PlaceID, Restaurant_Type, Resturant_Name, City, State, latitude, longitude, URL)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",(RestaurantID,current_datetime,username,place_id,Restaurant_Type,selected_restaurant,selected_city,state,lat,lng,url))
    
    
    print("Selected restaurant inserted into the database.")

main()
