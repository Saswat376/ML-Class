# This is for web scrapping--

import requests
from bs4 import BeautifulSoup

def scrape_product_reviews(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    reviews = []
    
    # Finding all review containers 
    review_containers = soup.find_all('div', class_='review-item') 
    
    for container in review_containers:
        # Extracting the review text
        review_text = container.find('p', class_='review-text').text.strip()
        # Extracting the star rating
        rating_tag = container.find('span', class_='star-rating') 
        rating = rating_tag['data-rating'] if rating_tag else None

        reviews.append({'text': review_text, 'rating': rating})

    return reviews
