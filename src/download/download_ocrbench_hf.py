import os
from datasets import load_dataset

def download_ocrbench():
    dataset_name = "echo840/OCRBench"
    save_dir = "./data/OCRBench_HF"
    
    print(f"Starting download of {dataset_name} from Hugging Face...")
    
    # 使用 HF 镜像加速（如果适用）
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    
    try:
        # OCRBench 通常很大，包含图片。我们先加载并保存到本地磁盘。
        # split="test" 假设通常评测集都在 test split
        ds = load_dataset(dataset_name, split="test")
        
        print(f"Dataset downloaded. Size: {len(ds)}")
        print(f"First sample keys: {ds[0].keys()}")
        
        # 保存到本地，方便后续评测脚本读取，无需每次联网
        ds.save_to_disk(save_dir)
        print(f"Dataset saved to {save_dir}")
        
    except Exception as e:
        print(f"Error downloading dataset: {e}")

if __name__ == "__main__":
    download_ocrbench()
