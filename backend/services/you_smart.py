import os
import requests
import json
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("YDC_API_KEY") or os.getenv("X-API-Key")

def call_smart(prompt: str, use_tools: bool = False):
    """Call You.com Advanced Agent API for intelligent reasoning and analysis.
    
    Args:
        prompt: The input prompt for the agent
        use_tools: Whether to enable research and compute tools (slower but more capable)
    """
    if not API_KEY:
        return "API key not configured. Please set YDC_API_KEY or X-API-Key environment variable."
    
    # Try with tools first, then fallback to no tools if it times out
    result = _make_api_request(prompt, use_tools, API_KEY)
    
    # If tools failed due to timeout, try again without tools
    if use_tools and "Request failed:" in result and "timed out" in result:
        print("⚠️ Advanced tools timed out, retrying without tools...")
        result = _make_api_request(prompt, False, API_KEY)
        if "Request failed:" not in result:
            result = "⚡ " + result + "\n\n*Note: Answered quickly without advanced analysis tools due to processing time limits.*"
    
    return result

def _make_api_request(prompt: str, use_tools: bool, api_key: str) -> str:
    """Make the actual API request"""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    # Use the Advanced Agent API endpoint
    url = "https://api.you.com/v1/agents/runs"
    
    payload = {
        "agent": "advanced",
        "input": prompt
    }
    
    # Add tools if requested (makes responses slower but more capable)
    if use_tools:
        payload["tools"] = [
            {"type": "research"},
            {"type": "compute"}
        ]

    try:
        # Set timeout based on whether tools are enabled - shorter timeout to fail faster
        timeout = 40 if use_tools else 10
        response = requests.post(url, headers=headers, json=payload, timeout=timeout)
        
        if response.status_code == 200:
            data = response.json()
            # Extract response from Advanced Agent API format
            if "output" in data:
                return data["output"]
            elif "result" in data:
                if isinstance(data["result"], str):
                    return data["result"]
                elif isinstance(data["result"], dict) and "output" in data["result"]:
                    return data["result"]["output"]
            elif "response" in data:
                return data["response"]
            elif "answer" in data:
                return data["answer"]
            else:
                return str(data)
        else:
            return f"API Error {response.status_code}: {response.text}"
            
    except requests.exceptions.RequestException as e:
        return f"Request failed: {str(e)}"
    except json.JSONDecodeError as e:
        return f"JSON decode error: {str(e)}"
    except Exception as e:
        return f"Unexpected error: {str(e)}"