import os
import sys
from modelscope import snapshot_download
from modelscope.hub.api import HubApi

def download_models():
    # 你的 ModelScope Token
    token = "ms-56df1b79-fe15-4dd6-911a-5219b0a6633a"
    
    # 要下载的模型列表
    models = [
        "Qwen/Qwen3-VL-4B-Instruct",
        "Qwen/Qwen3-VL-2B-Instruct"
    ]

    print(f"Logging in with provided token...")
    try:
        api = HubApi()
        api.login(token)
        print("Login successful.")
    except Exception as e:
        print(f"Login failed: {e}")
        # 继续尝试下载，因为有些模型可能是公开的

    for model_id in models:
        print(f"\n{'='*50}")
        print(f"Start downloading: {model_id}")
        print(f"{'='*50}")
        
        try:
            # 下载到当前目录 (会自动创建 Qwen/Qwen3-VL-4B-Instruct 这样的结构)
            model_dir = snapshot_download(model_id, cache_dir='.')
            print(f"\n✅ Successfully downloaded: {model_id}")
            print(f"Local path: {model_dir}")
        except Exception as e:
            print(f"\n❌ Failed to download {model_id}: {e}")

if __name__ == "__main__":
    download_models()
