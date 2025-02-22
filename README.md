# Date-Recommender
## Abstact
The Date-Recommender is a Python script that connects to a Duck database, retrieves data, performs natural language processing (NLP), trains a machine learning model, and utilizes the Google Places API to fetch data about restaurants and bars. It then saves the processed data into JSON files and converts it into a DataFrame, then uploads the selected restaurant back into the database.

## Introduction
The Date-Recommender is a tool designed for individuals who struggle with indecisiveness when choosing places to visit in a city. This model utilizes your previous ratings of places you've visited to generate personalized recommendations for new places to explore. The recommendations are based on your preferences and ratings, helping you discover exciting locations that align with your interests in a new city.

## Prerequisites
To run the Date-Recommender, you'll need the following dependencies:

- Python 3.x
- MySQL Connector/Python
- Google Maps API Key

## Installation
1. Clone the repository to your local machine.

2. Install the required Python packages using the following command:

pip install -r requirements.txt

3. Set up a MySQL database to store the information about the places you've visited and your ratings. Ensure you have the necessary credentials (host, username, password) to connect to the database.

4. Obtain a Google Maps API Key by following the instructions provided by Google.

5. Update the configuration file config.py with your MySQL database credentials and Google Maps API Key.
https://developers.google.com/maps/documentation/places/web-service/get-api-key

## Usage
1. Add places you've visited and their descriptions and rate them on a scale from 1 to 10 in the MySQL database. Use a separate file or import them directly into the database. (instructions in database creation folder, I created a separate CSV and inserted into database)

2. Launch the Date-Recommender application by running the following command:

python date_recommender.py

3. Follow the prompts to select your preferences and options for recommendations.

4. The Date-Recommender will retrieve your previous ratings from the database, perform NLP preprocessing, train a machine learning model, and generate a list of recommended places for you to visit in the city.

5. Review the recommendations and select a place that interests you.

6. Enjoy your date at the recommended location!

## Configuration
Update the database section with the necessary configuration details:

- DB_HOST: The host address of your MySQL database.
- DB_USERNAME: The username to connect to the MySQL database.
- DB_PASSWORD: The password for the MySQL database.
- DB_DATABASE: The name of the MySQL database.
- GOOGLE_MAPS_API_KEY: Your Google Maps API Key.

## Troubleshooting
If you encounter any issues while using the Date-Recommender, consider the following troubleshooting steps:

- Ensure that your MySQL database is properly set up and running.
- Verify that the database connection details in the database section are correct.
- Double-check the installation of the required dependencies and packages.
- Make sure you have a valid and active Google Maps API Key.
- If the issue persists, feel free to reach out to the developer for assistance.


## PowerBI Dashboard I created to visualize KPIs.
![image (1)](https://github.com/Tartticus/Date-Recommender-/assets/87999009/68094090-4d48-4048-be7d-a75ee17a873d)


## Contributing
Contributions to the Date-Recommender project are welcome!
