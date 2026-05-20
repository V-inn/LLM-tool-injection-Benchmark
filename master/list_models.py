import os
from google import genai
from dotenv import load_dotenv

def list_available_models():
    print("========================================")
    print(" [*] GOOGLE AI CLOUD: MODEL DIAGNOSTIC")
    print("========================================")
    
    # Load the GEMINI_API_KEY from .env
    load_dotenv()
    
    try:
        # Initialize the new SDK client
        client = genai.Client()
        print("[+] Authentication successful. Fetching available models...\n")
        
        # Retrieve the list of models available to this specific API Key
        models = client.models.list()
        
        count = 0
        for model in models:
            # We filter for models that support text/content generation
            if "generateContent" in getattr(model, "supported_actions", []):
                print(f"  -> {model.name}")
                count += 1
                
        # Fallback if the new SDK structure omits supported_actions
        if count == 0:
            for model in client.models.list():
                print(f"  -> {model.name}")
                count += 1
                
        print(f"\n[+] Total text models available: {count}")
        
    except Exception as e:
        print(f"\n[-] Critical Connection Error: {e}")

if __name__ == "__main__":
    list_available_models()