import json
import numpy as np
import requests
import nltk
import googlemaps
import PySimpleGUI as sg
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
print("imports complete")

con = duckdb.connect(database='dining')

#%% Setup


#%% Select
options = ['Dining','Dessert','Japanese','Ramen','Taiwanese','Bar','Italian']


    # Print available options
print("Please choose one of the following options:")
for i, option in enumerate(options, 1):
    print(f"{i}. {option}")
while True:   
    # Prompt the user for input
    choice = input("Enter the number corresponding to your choice: ")

    # Validate the user's input
    if not choice.isdigit() or int(choice) < 1 or int(choice) > len(options):
        print("Invalid choice. Please try again.")
    
    else:
        selected_option = options[int(choice) - 1]
        print(f"You selected: {selected_option}")    
        break







#%% select 2

    
#%% select statement 

start = time.time()

exclude_df = con.execute(f"SELECT * from Restaurants").df()
df = con.execute(f"SELECT Description,  CASE WHEN UserRating >= 7 THEN 'High' WHEN UserRating >= 4 THEN 'Medium' ELSE 'Low' END AS score FROM Restaurants WHERE Type = '{selected_option}'").df()



end = time.time()
elapsed = end - start
print(f"this query execution took {elapsed: .2f} seconds")    

#%%

# Map score values to 1, 2, 3
score_mapping = {'High': 1, 'Medium': 2, 'Low': 3}
df['score'] = df['score'].map(score_mapping)
print(df)
df_string = df.to_string(index=False)
print(df_string)

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

# Apply preprocessing to the 'description' column
df['processed_description'] = df['Description'].apply(preprocess_text)

print(df)


#%% Machine learning

# Create the CountVectorizer
vectorizer = CountVectorizer()

# Encode the preprocessed text data
X = vectorizer.fit_transform(df['processed_description'])
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




#%% API 
# Replace 'YOUR_API_KEY' with your actual API key
api_key ='AIzaSyB6HqyD1tYpWOjwswMN95hC_qvSdrgJVJU'
gmaps = googlemaps.Client(key=api_key)

#user input for cities
cities = ['Los Angeles', 'Pasadena', 'Arcadia', 'Santa Monica', 'Glendale','San Diego']


   # Print available options
print("Please choose one of the following options:")
for i, city in enumerate(cities, 1):
    print(f"{i}. {city}")

while True:
    # Prompt the user for input
    choice2 = input("Enter the number corresponding to your choice: ")

    try:
        # Validate the user's input
        choice_index = int(choice2) - 1

        if choice_index < 0 or choice_index >= len(cities):
            print("Invalid choice. Please try again.")
        else:
            selected_city = cities[choice_index]
            print(f"You selected: {selected_city}")

            # Convert the selected city to coordinates
            geocode_result = gmaps.geocode(selected_city)
            if geocode_result:
                lat = geocode_result[0]['geometry']['location']['lat']
                lng = geocode_result[0]['geometry']['location']['lng']
                print(f"Coordinates: {lat}, {lng}")
            
            else:
                print("Failed to retrieve coordinates for the selected city.")
            break
    except ValueError:
        print("Invalid choice. Please enter a number.")
    
    

#%%city

# Extract the restaurant names to exclude
excluded_restaurants = exclude_df['Username'].tolist()

# Specify the state of California as the location
selected_state = 'California'

# Perform the place search in the selected city and state
places_result = gmaps.places(query=f'{selected_city}, {selected_state}')

# Search for restaurants in the selected city and state
restaurants_result = gmaps.places_nearby(type='restaurant', location= f"{lat},{lng}" , radius = 11519, open_now = True)

# Search for bars in the selected city and state
bars_result = gmaps.places_nearby(type='bar', location= f"{lat},{lng}" , radius = 11519, open_now = True)

# Extract restaurant information
restaurants = restaurants_result['results']
restaurant_data = []
for restaurant in restaurants:
    place_id = restaurant['place_id']
    place_details = gmaps.place(place_id=place_id, fields=['name', 'vicinity', 'reviews'])
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
        'reviews': review_data
    })

# Extract bar information
bars = bars_result['results']
bar_data = []
for bar in bars:
    place_id = bar['place_id']
    place_details = gmaps.place(place_id=place_id, fields=['name', 'vicinity', 'reviews'])
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
        'reviews': review_data
    })

# Save the extracted data to a JSON file
output_file = 'places_data.json'
data = {
    'restaurants': restaurant_data,
    'bars': bar_data
}
with open(output_file, 'w') as json_file:
    json.dump(data, json_file, indent=4)

print(f"Data saved to {output_file}")

#%% Json processing (first checks for high score - then medium)
model = naive_bayes

# Load the JSON data
with open('places_data.json', 'r') as json_file:
    data = json.load(json_file)

restaurants_data = data['restaurants']

# Preprocess the restaurant reviews
preprocessed_reviews = []
for restaurant in restaurants_data:
    reviews = restaurant['reviews']
    preprocessed_reviews.extend([preprocess_text(review) for review in reviews])

# Vectorize the preprocessed reviews
X_test = vectorizer.transform(preprocessed_reviews)

# Make predictions using the trained model
y_pred = model.predict(X_test)

# Map the predicted scores to labels
score_labels = {1: 'High', 2: 'Medium', 3: 'Low'}
predicted_scores = [score_labels[pred] for pred in y_pred]

# Update the restaurant data with predicted scores
for i, restaurant in enumerate(restaurants_data):
    restaurant['predicted score'] = predicted_scores[i]

# Create the classified data
classified_data = {
    'restaurants': restaurants_data,
    'bars': data['bars']
}

# Save the updated data to a new JSON file
output_file = 'classified_places_data.json'
with open(output_file, 'w') as json_file:
    json.dump(classified_data, json_file, indent=4)
 
print(f"Data saved to {output_file}")


#%% convert to dataframe

# Load the JSON data
with open('classified_places_data.json', 'r') as json_file:
    classified_data = json.load(json_file)

restaurants_data = classified_data['restaurants']

# Create a list of dictionaries to store the restaurant data
restaurant_list = []
for restaurant in restaurants_data:
    name = restaurant['name']
    score = restaurant.get('predicted score')
    if score:  # Only include restaurants with a predicted score
        restaurant_list.append({'name': name, 'predicted score': score})

# Convert the list of dictionaries to a DataFrame
df = pd.DataFrame(restaurant_list)

# Print the DataFrame
print(df)

#%% Restaurant selection and upload
# Prompt the user to pick a restaurant from the DataFrame
print("Please choose a restaurant from the following options:")
for i, restaurant in enumerate(df['name'], 1):
    print(f"{i}. {restaurant}")

while True:
    # Prompt the user for input
    choice = input("Enter the number corresponding to your choice: ")

    try:
        # Validate the user's input
        choice_index = int(choice) - 1

        if choice_index < 0 or choice_index >= len(df['name']):
            print("Invalid choice. Please try again.")
        else:
            selected_restaurant = df['name'][choice_index]
            print(f"You selected: {selected_restaurant}")
            break
    except ValueError:
        print("Invalid choice. Please enter a number.")

# Get the current date and time
current_datetime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Prepare the INSERT statement
con.execute("INSERT INTO selected_restaurants (name, selected_datetime) VALUES (%s, %s)")



print("Selected restaurant inserted into the database.")
