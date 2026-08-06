import os
import requests
import logging
from typing import List, Tuple
from urllib.parse import urlparse, urlunparse
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

# Load Apify token
APIFY_API_TOKEN = os.getenv("APIFY_API_TOKEN")

# Pre-defined mock reviews for fallback mode (so the presentation never fails)
MOCK_BACKPACK_REVIEWS = [
    "I bought this backpack for college. It has a ton of space and easily fits my 15-inch laptop, textbooks, and water bottle. Highly recommended!",
    "The zipper broke on the second day of using it. It's a nice looking bag, but the build quality is extremely cheap and flimsy.",
    "Very comfortable straps. Even when it's fully loaded with books, it doesn't hurt my shoulders. The design is simple and clean.",
    "Decent bag for the price. Not fully waterproof, but handles light rain okay. It could use a few more internal organizing pockets.",
    "Material feels rough and the stitching is already coming loose near the top handle. Returning it today."
]

MOCK_TABLET_REVIEWS = [
    "The screen is gorgeous. Bright colors and very sharp for watching Netflix. Battery easily lasts all day on a single charge.",
    "Extremely slow. Apps lag and web pages take forever to load. The operating system is clunky and frustrating to use.",
    "Perfect first tablet for my kid. The parental controls are excellent and it has a nice durable build. Good value for money.",
    "Audio quality is surprisingly good. But it is very heavy to hold for long reading sessions and the charger cable is too short.",
    "The Wi-Fi connection drops constantly. I have to restart the tablet every hour just to keep it online. Very disappointed."
]

MOCK_GENERAL_REVIEWS = [
    "Exactly as described! Excellent product, shipped very fast, and works perfectly out of the box.",
    "It works okay, but the price is too high for what you get. Just an average product with nothing special.",
    "Complete waste of money. It arrived broken and customer support has been ignoring my return requests.",
    "Very pleased with the purchase. High quality material and looks very elegant. Will buy again.",
    "The battery dies within 2 hours. It's almost useless unless you keep it plugged into the wall constantly."
]

def get_fallback_reviews(url: str) -> List[str]:
    """
    Returns relevant mock reviews based on keywords in the product URL.
    
    Args:
        url (str): The product URL string.
        
    Returns:
        list: A list of pre-defined mock review strings.
    """
    url_lower = url.lower()
    if "backpack" in url_lower or "bag" in url_lower:
        return MOCK_BACKPACK_REVIEWS
    elif "tablet" in url_lower or "kindle" in url_lower or "fire" in url_lower:
        return MOCK_TABLET_REVIEWS
    else:
        return MOCK_GENERAL_REVIEWS

def resolve_url(url: str) -> str:
    """
    Follows HTTP redirects to expand shortened Amazon links (like amzn.in or a.co)
    to full e-commerce product links that the scraper can parse.
    
    Args:
        url (str): The shortened Amazon link.
        
    Returns:
        str: The resolved, expanded destination URL.
    """
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}
        response = requests.head(url, allow_redirects=True, timeout=5, headers=headers)
        if response.status_code >= 400:
            response = requests.get(url, allow_redirects=True, timeout=5, headers=headers, stream=True)
        return response.url
    except Exception as e:
        logger.warning(f"Redirect resolution failed for {url}: {str(e)}")
        return url

def clean_amazon_url(url: str) -> str:
    """
    Cleans an Amazon URL by stripping query parameters and tracking hashes.
    
    Args:
        url (str): The expanded Amazon URL.
        
    Returns:
        str: The clean URL containing only the base scheme, host, and product path.
    """
    try:
        parsed = urlparse(url)
        # Keep only scheme, netloc, and path (remove query params)
        cleaned = urlunparse((parsed.scheme, parsed.netloc, parsed.path, '', '', ''))
        return cleaned
    except Exception as e:
        logger.warning(f"Failed to clean Amazon URL {url}: {str(e)}")
        return url

def scrape_amazon_reviews(url: str, max_reviews: int = 10) -> Tuple[List[str], bool]:
    """
    Scrapes reviews from Amazon using the Apify cloud actor.
    Falls back to mock reviews if API token is missing or if the API call fails.
    
    Args:
        url (str): Product link or shortened share URL.
        max_reviews (int): Maximum number of reviews to extract.
        
    Returns:
        tuple: (list of reviews, is_simulated_boolean)
    """
    # 1. Resolve and clean URL format first
    resolved_url = resolve_url(url)
    cleaned_url = clean_amazon_url(resolved_url)
    
    logger.info(f"Raw URL: {url}")
    logger.info(f"Resolved URL: {resolved_url}")
    logger.info(f"Cleaned URL for Scraper: {cleaned_url}")
        
    # Check if API token is valid
    if not APIFY_API_TOKEN or APIFY_API_TOKEN == "your_apify_api_token_here":
        logger.warning("Apify API Token not found. Activating fallback simulator mode.")
        return get_fallback_reviews(cleaned_url), True

    try:
        from apify_client import ApifyClient
        
        logger.info(f"Initializing Apify Client to scrape: {cleaned_url}")
        client = ApifyClient(APIFY_API_TOKEN)
        
        run_input = {
            "productUrls": [{"url": cleaned_url}],
            "maxReviews": max_reviews,
            "sort": "helpful",
            "filterByRating": "all"
        }
        
        # Call the Amazon Reviews Scraper actor on Apify
        run = client.actor("junglee/amazon-reviews-scraper").call(run_input=run_input)
        
        # Fetch items from the default run dataset
        dataset_items = client.dataset(run.default_dataset_id).list_items().items
        
        reviews = []
        for item in dataset_items:
            review_text = item.get("reviewDescription")
            if review_text:
                reviews.append(review_text)
                
        if not reviews:
            logger.warning("Scraper completed but returned empty reviews list. Activating fallback.")
            return get_fallback_reviews(cleaned_url), True
            
        logger.info(f"Successfully scraped {len(reviews)} real reviews from Apify.")
        return reviews, False
        
    except Exception as e:
        logger.error(f"Apify call failed: {str(e)}. Activating fallback simulator.")
        return get_fallback_reviews(cleaned_url), True
