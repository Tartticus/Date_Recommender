def insert_place_chosen(selected_restaurant,city):
    query = f"{selected_restaurant}, {city}"
    
    find_response = gmaps.find_place(
        input=query,
        input_type="textquery",
        fields=['place_id', 'name']
    )
    candidates = find_response.get('candidates', [])
    if not candidates:
        print(f"No results found for: {query}")
        

    # Take the first candidate from the results.
    candidate = candidates[0]
    place_id = candidate.get('place_id')
    
    # Now retrieve detailed information using the Place Details API.
    details_response = gmaps.place(
        place_id=place_id,
        fields=['name', 'formatted_address', 'reviews', 'url']
    )

    details = details_response.get('result', {})
    url = details.get("url")
    return place_id, url
