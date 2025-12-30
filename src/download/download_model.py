import os
import sys
from modelscope import snapshot_download
from modelscope.hub.api import HubApi

def download_private_model():
    model_id = "JunfengPan/Qwen3-VL-8B-Instruct-AWQ"
    # HARDCODED TOKEN AS REQUESTED
    token = "ms-56df1b79-fe15-4dd6-911a-5219b0a6633a"
    
    print(f"Logging in with provided token...")
    try:
        api = HubApi()
        api.login(token)
        print("Login successful.")
    except Exception as e:
        print(f"Login failed: {e}")
        sys.exit(1)

    print(f"Downloading model: {model_id} to current directory...")
    try:
        # cache_dir='.' will download to ./JunfengPan/Qwen3-VL-8B-Instruct-AWQ
        model_dir = snapshot_download(model_id, cache_dir='.')
        
        print("\n✅ Model downloaded successfully!")
        print(f"Model path: {model_dir}")
            
    except Exception as e:
        print(f"\n❌ Download failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    download_private_model()