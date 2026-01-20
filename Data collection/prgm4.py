# This is for API collection and saving to a CSV file.

import requests
import pandas as pd
import os

def fetch_and_save_posts(folder='posts_data', filename='posts.csv'):
    """
    Fetches posts from the JSONPlaceholder API and saves them to a CSV file.
    """
    # Create the directory if it doesn't exist
    os.makedirs(folder, exist_ok=True)
    
    # API endpoint for posts
    url = "https://jsonplaceholder.typicode.com/posts"
    
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad status codes
        
        # Convert the JSON response to a pandas DataFrame
        posts_df = pd.DataFrame(response.json())
        
        # Save the DataFrame to a CSV file
        file_path = os.path.join(folder, filename)
        posts_df.to_csv(file_path, index=False)
        
        print(f"Successfully fetched and saved {len(posts_df)} posts to '{file_path}'")
        
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")

if __name__ == "__main__":
    fetch_and_save_posts()
