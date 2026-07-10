
import logging
from app.router import build_from_config
from app.config import iter_api_keys, iter_models, ACTIVE_PROVIDER

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_gemini_connection():
    print(f"Active Provider: {ACTIVE_PROVIDER}")
    
    try:
        # We use build_from_config to create a router based on current env
        router = build_from_config(iter_api_keys(), iter_models())
        
        messages = [{"role": "user", "content": "Hello! Are you working?"}]
        print("Sending test message to Gemini...")
        
        response = router.chat(messages)
        print("--- Response ---")
        print(response)
        print("----------------")
        print("SUCCESS: Gemini API is working!")
        
    except Exception as e:
        print("FAILURE:")
        print(e)
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_gemini_connection()
