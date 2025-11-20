"""
Article Rectification System

This is where you implement your article rectification logic.
The run() function receives AI-generated content and should return the corrected version.

Feel free to:
- Add additional modules, classes, or helper functions
- Load and compare with source articles
- Implement multi-step validation and correction strategies
- Use multiple LLM calls or different models
- Add confidence scoring and logging
"""

from dotenv import load_dotenv
from litellm import completion
import os

load_dotenv()

def run(ai_generated_content: str) -> str:
    """
    Rectify an AI-generated article.
    
    Args:
        ai_generated_content: The AI-generated article text to be corrected
        
    Returns:
        str: The rectified article content
    """
    # Create a simple prompt to fix issues
    prompt = (
        "Fix all issues in the following article:\n\n"
        f"{ai_generated_content}\n\n"
        
        "Return only the corrected article text."
    )

    # Call LLM to rectify the article
    response = completion(
        model="openai/devbot/gpt-oss-120b",
        messages=[
            {"role": "user", "content": prompt}
        ],
        api_key=os.getenv('LLM_API_KEY'),
        api_base=os.getenv('LLM_API_BASE')
    )
    
    rectified_content = response.choices[0].message.content.strip()
    return rectified_content    

