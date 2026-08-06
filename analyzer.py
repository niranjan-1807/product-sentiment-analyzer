import os
import json
import logging
from typing import Dict, Any, Optional
from dotenv import load_dotenv
from groq import Groq

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

# Let the model be configured via environment variables with a stable fallback
MODEL_NAME = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

api_key = os.getenv("GROQ_API_KEY")
client = Groq(api_key=api_key) if api_key else None

SYSTEM_PROMPT = """
You are an expert product review analyst. Your task is to perform open aspect-based sentiment analysis on customer product reviews.
You must analyze the review text, identify which specific aspects, features, or parts of the product are mentioned (e.g., battery life, price, zipper quality, screen, comfort, packaging, shipping speed), and determine the sentiment for each of those aspects.

You must output a JSON object matching this schema:
{
  "sentiment": "Positive" | "Neutral" | "Negative",
  "confidence": float (between 0.0 and 1.0),
  "reasoning": "A concise explanation of your sentiment choice.",
  "aspects": [
    {
      "aspect": "name of the aspect/feature identified in the review (e.g., screen, zipper, price, battery life)",
      "sentiment": "Positive" | "Neutral" | "Negative"
    }
  ]
}

Ensure your response is valid JSON and contains NO conversational text before or after the JSON.
If no specific product features or aspects are mentioned in the review, return an empty list for "aspects".
"""

# Few-shot examples demonstrating how to dynamically extract aspects and sentiments
FEW_SHOT_EXAMPLES = [
    {
        "review": "Purchased this on sale. Great price and amazing sound. It is well built, but you have to pay $15 to remove ads.",
        "response": {
            "sentiment": "Positive",
            "confidence": 0.90,
            "reasoning": "The review highlights great price, amazing sound, and durability, which outweigh the minor annoyance of paying to remove ads.",
            "aspects": [
                {"aspect": "price", "sentiment": "Positive"},
                {"aspect": "sound quality", "sentiment": "Positive"},
                {"aspect": "build quality", "sentiment": "Positive"},
                {"aspect": "ads", "sentiment": "Negative"}
            ]
        }
    },
    {
        "review": "Just an average tablet. Does show a few things on screen but still limited.",
        "response": {
            "sentiment": "Neutral",
            "confidence": 0.85,
            "reasoning": "The reviewer describes the product as average and limited, expressing neither strong satisfaction nor dissatisfaction.",
            "aspects": [
                {"aspect": "overall quality", "sentiment": "Neutral"},
                {"aspect": "screen display", "sentiment": "Neutral"}
            ]
        }
    },
    {
        "review": "Was cheap, but returned it. Cannot run basic apps, and connection drops constantly.",
        "response": {
            "sentiment": "Negative",
            "confidence": 0.95,
            "reasoning": "The product was returned due to lack of performance, inability to run apps, and connectivity failure.",
            "aspects": [
                {"aspect": "price", "sentiment": "Positive"},
                {"aspect": "app compatibility", "sentiment": "Negative"},
                {"aspect": "connectivity", "sentiment": "Negative"}
            ]
        }
    }
]

def analyze_review(review_text: str, custom_guidelines: Optional[str] = None) -> Dict[str, Any]:
    """
    Analyzes a product review using Llama 3 model family on Groq, performing
    Open Aspect-Based Sentiment Analysis (ABSA) and returning a structured JSON result.
    
    Args:
        review_text (str): The raw text of the review to analyze.
        custom_guidelines (str, optional): Custom execution constraints provided by the user.
        
    Returns:
        dict: A dictionary containing overall sentiment, confidence score,
              reasoning explanation, and a list of extracted aspect sentiments.
    """
    if not client:
        logger.warning("Groq API client is not initialized. Check GROQ_API_KEY environment variable.")
        return {
            "sentiment": "Neutral",
            "confidence": 0.0,
            "reasoning": "Groq API client is not initialized. Please verify your GROQ_API_KEY in the .env file.",
            "aspects": []
        }
        
    system_instruction = SYSTEM_PROMPT
    if custom_guidelines:
        system_instruction += f"\nCustom User Guidelines: {custom_guidelines}"

    messages = [{"role": "system", "content": system_instruction}]
    
    # Add few-shot examples
    for ex in FEW_SHOT_EXAMPLES:
        messages.append({"role": "user", "content": f"Review: {ex['review']}"})
        messages.append({"role": "assistant", "content": json.dumps(ex['response'])})
    
    messages.append({"role": "user", "content": f"Review: {review_text}"})

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0.2 # Lower temperature for deterministic classification
        )
        raw_json = response.choices[0].message.content
        return json.loads(raw_json)
    except Exception as e:
        logger.error(f"Groq API call failed: {str(e)}")
        return {
            "sentiment": "Neutral",
            "confidence": 0.0,
            "reasoning": f"Error running analysis: {str(e)}",
            "aspects": []
        }

if __name__ == "__main__":
    test_review = "I hate the battery life. It drains in an hour. But screen is bright."
    logger.info("Analyzing sample review...")
    res = analyze_review(test_review)
    print(json.dumps(res, indent=2))