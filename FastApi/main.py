from fastapi import FastAPI, HTTPException, Request
import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import re
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from collections import Counter
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import pipeline
import requests

app = FastAPI()

# CORS configuration (optional, remove if not needed)
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust as needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()
transformer_sentiment = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

def analyze_review(text):
    score = sia.polarity_scores(text)
    return {
        'sentiment': score,
        'is_fake': score['compound'] < -0.5  # Example logic for identifying fake reviews
    }

def extract_amazon_reviews(product_url, max_pages=5):
    product_id = re.search(r'/dp/([A-Z0-9]{10})|/([A-Z0-9]{10})/', product_url)
    if not product_id:
        print("Invalid Amazon product URL.")
        return []

    product_id = product_id.group(1) or product_id.group(2)
    review_url = f"https://www.amazon.in/product-reviews/{product_id}/ref=cm_cr_getr_d_paging_btm_next_1?ie=UTF8&reviewerType=all_reviews&sortBy=recent&pageNumber=1"
    all_reviews = []

    for page in range(1, max_pages + 1):
        current_review_url = f"{review_url}&pageNumber={page}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.93 Safari/537.36"
        }

        response = requests.get(current_review_url, headers=headers)

        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            reviews = soup.find_all('div', {'data-hook': 'review'})

            for review in reviews:
                text = review.find('span', {'data-hook': 'review-body'}).text.strip().replace('\n', ' ')
                if text:  # Only add non-empty reviews
                    all_reviews.append({'Text': text})

            print(f"Extracted {len(reviews)} reviews from Amazon page {page}...")
            time.sleep(1)  # Reduced sleep time for efficiency
        else:
            print(f"Failed to retrieve Amazon page {page}. Status code: {response.status_code}")

    return all_reviews

def extract_flipkart_reviews(product_url, max_pages=3):
    driver = webdriver.Chrome()  # Ensure you have the correct driver installed
    all_reviews = []
    
    try:
        # Navigate to the main product page
        driver.get(product_url)
        print("Navigated to the product page.")

        # Construct the reviews URL
        reviews_url = product_url.replace('/p/', '/product-reviews/') + '&page=1'
        
        for page in range(1, max_pages + 1):
            current_page_url = reviews_url.replace('&page=1', f'&page={page}')
            driver.get(current_page_url)
            print(f"Extracting reviews from Flipkart page {page}...")

            WebDriverWait(driver, 10).until(
                EC.presence_of_all_elements_located((By.XPATH, '//*[@id="container"]/div/div[3]/div/div/div[2]/div'))
            )
            
            reviews = driver.find_elements(By.XPATH, '//*[@id="container"]/div/div[3]/div/div/div[2]/div')
            page_reviews = [{'Text': review.text.strip().replace('\n', ' ')} for review in reviews if review.text.strip()]

            all_reviews.extend(page_reviews)
            print(f"Extracted {len(page_reviews)} reviews from page {page}.")

    except Exception as e:
        print(f"Error while extracting reviews: {e}")
    finally:
        driver.quit()

    print(f"Total extracted reviews: {len(all_reviews)}.")
    return all_reviews


def extract_bookmyshow_reviews_selenium(product_url):
    driver = webdriver.Chrome()  # Ensure you have the correct driver installed
    all_reviews = []

    try:
        driver.get(product_url)
        # Wait for the review elements to load
        WebDriverWait(driver, 10).until(
            EC.presence_of_all_elements_located((By.CLASS_NAME, 'sc-r6zm4d-14'))  # Update this if necessary
        )
        
        reviews = driver.find_elements(By.CLASS_NAME, 'sc-r6zm4d-14')
        all_reviews = [{'Text': review.text.strip().replace('\n', ' ')} for review in reviews if review.text.strip()]

        print(f"Extracted {len(all_reviews)} reviews from BookMyShow...")
    except Exception as e:
        print(f"Error while extracting reviews: {e}")
    finally:
        driver.quit()

    return all_reviews

def extract_reviews(product_url, max_pages=5):
    if "amazon.in" in product_url:
        return extract_amazon_reviews(product_url, max_pages)
    elif "flipkart.com" in product_url:
        return extract_flipkart_reviews(product_url, max_pages)
    elif "bookmyshow.com" in product_url:
        return extract_bookmyshow_reviews_selenium(product_url)
    else:
        print("Unsupported URL.")
        return []

def analyze_sentiment(review_text):
    vader_score = sia.polarity_scores(review_text)
    vader_sentiment = 'neutral'
    if vader_score['compound'] >= 0.50:
        vader_sentiment = 'positive'
    elif vader_score['compound'] <= -0.50:
        vader_sentiment = 'negative'

    transformer_result = transformer_sentiment(review_text)[0]
    transformer_sentiment_label = transformer_result['label']
    transformer_score = transformer_result['score']

    return {
        "vader_score": vader_score,
        "vader_sentiment": vader_sentiment,
        "transformer_sentiment": transformer_sentiment_label,
        "transformer_score": transformer_score,
        "flagged": abs(vader_score['compound']) > 0.75 or transformer_score > 0.75
    }

# Linguistic Pattern Analysis Function (with N-Gram Analysis)
def analyze_linguistic_patterns(review_text, n=3):
    punctuation_count = sum(1 for char in review_text if char in ['!', '?'])
    words = re.findall(r'\w+', review_text.lower())
    word_counts = Counter(words)

    repeated_words = [word for word, count in word_counts.items() if count > 2]

    ngrams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
    ngram_counts = Counter(ngrams)
    suspicious_ngrams = [ngram for ngram, count in ngram_counts.items() if count > 1]

    suspicion_score = punctuation_count + len(repeated_words) + len(suspicious_ngrams)

    return {
        "suspicion_score": suspicion_score,
        "repeated_words": repeated_words,
        "excessive_punctuation": punctuation_count > 3,
        "suspicious_ngrams": suspicious_ngrams
    }

# Reviewer Behavior Analysis Function
def analyze_reviewer_behavior(review_text):
    flags = {
        "length_check": len(review_text) < 30,
        "unusual_length": len(review_text) > 1000
    }
    return flags

# Grammar and Spelling Analysis Function
def grammar_check(review_text):
    url = "https://api.languagetoolplus.com/v2/check"
    params = {
        'text': review_text,
        'language': 'en-US'
    }
    response = requests.post(url, data=params)

    if response.status_code == 200:
        matches = response.json().get('matches', [])
        error_count = len(matches)
        return {"grammar_issues": error_count, "flagged": error_count > 2}
    else:
        print("Error in grammar checking:", response.status_code)
        return {"grammar_issues": 0, "flagged": False}


# Combined Review Analysis Function
def analyze_review(review_text):
    sentiment_result = analyze_sentiment(review_text)
    linguistic_result = analyze_linguistic_patterns(review_text)
    behavior_result = analyze_reviewer_behavior(review_text)
    grammar_result = grammar_check(review_text)

    flags_count = sum([
        sentiment_result['flagged'],
        linguistic_result['suspicion_score'] > 1,
        behavior_result['length_check'],
        grammar_result['flagged']
    ])

    is_fake = flags_count >= 2

    reasons = []
    if sentiment_result['flagged']:
        reasons.append("Extreme sentiment detected (VADER or Transformer).")
    if linguistic_result['suspicion_score'] > 1:
        reasons.append("High suspicion score due to linguistic patterns.")
    if behavior_result['length_check']:
        reasons.append("Review is unusually short.")
    if grammar_result['flagged']:
        reasons.append("Multiple grammar issues detected.")

    return {
        "sentiment": sentiment_result,
        "linguistic_patterns": linguistic_result,
        "behavior_flags": behavior_result,
        "grammar_flags": grammar_result,
        "is_fake": is_fake,
        "reasons": reasons
    }












#Code to write 
@app.get("/first")
def myFirstCode():
    return {"message":"hi sai charan are you feeling good"}

@app.get("/second")
def my_sec(str):
    return {"message": str }

@app.post("/review")
def main_method(INFO: dict):
    print(INFO["review"])
    user_input_url = INFO['review']
    all_reviews = extract_reviews(user_input_url)

    # Analyzing reviews
    if all_reviews:
        reviews_df = pd.DataFrame(all_reviews)
        reviews_df['Analysis'] = reviews_df['Text'].apply(analyze_review)
        reviews_df['Sentiment Score'] = reviews_df['Analysis'].apply(lambda x: x['sentiment']['compound'])

        fake_reviews = reviews_df[reviews_df['Analysis'].apply(lambda x: x['is_fake'])]
        fake_reviews = fake_reviews.drop_duplicates(subset=['Text'])
        fake_reviews_count = len(fake_reviews)
        total_reviews = len(reviews_df)
        fake_percentage = (fake_reviews_count / total_reviews * 100) if total_reviews > 0 else 0

        print(f"Percentage of Fake Reviews: {fake_percentage:.2f}%\n")

        flagged_fake_reviews = []

        if not fake_reviews.empty:
            for index, row in fake_reviews.iterrows():
                reasons = []
                analysis = row['Analysis']
                if analysis['sentiment']['compound'] < -0.5:
                    reasons.append("Extreme negative sentiment")
                if analysis['sentiment']['compound'] > 0.5:
                    reasons.append("Extreme positive sentiment")

                flagged_fake_reviews.append({
                    "index": index,
                    "review": row['Text'],
                    "reasons": reasons
                })

        return flagged_fake_reviews
    else:
        return []
    


@app.post("/reviews")
def main_method2 (INFO : dict):
    print("list")
    user_review = INFO["review"]
    analysis_result = analyze_review(user_review)

    # Create a dictionary for the output
    output = {
        "input-text": user_review,
        "sentiment score": analysis_result['sentiment']['vader_score']['compound'],  # Add sentiment score
        "Flagged as Fake": "Yes" if analysis_result['is_fake'] else "No",
        "Reasons for Flagging as Fake": []
    }

    if analysis_result['is_fake']:
        output["Reasons for Flagging as fake"] = analysis_result['reasons']

    # Output the dictionary
    print("\nAnalysis Output:")
    print(output)
    return output