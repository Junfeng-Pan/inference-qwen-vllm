import os
import requests
from PIL import Image
from io import BytesIO
from vllm import LLM, SamplingParams
from transformers import AutoProcessor

def run_inference():
    # 1. 获取模型路径
    model_dir = "./Qwen3-VL-8B-Instruct-AWQ"
    
    print(f"Loading model from: {model_dir}")

    # 2. 初始化 vLLM
    # Qwen3-VL (基于 Qwen2-VL 架构) 需要 trust_remote_code=True
    try:
        llm = LLM(
            model=model_dir,
            # quantization="awq", # Auto-detect from config (compressed-tensors)
            trust_remote_code=True,
            max_model_len=4096,
            limit_mm_per_prompt={"image": 1},
            enforce_eager=True, # 有时为了兼容性需要开启 eager 模式，可视情况调整
            gpu_memory_utilization=0.9,
        )
    except Exception as e:
        print(f"Failed to initialize vLLM: {e}")
        return

    # 3. 准备测试数据
    try:
        # 加载处理工具以正确构建 Prompt 格式
        processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
        
        # 下载示例图片
        # 使用 Qwen-VL 官方示例图片，或者一个稳定的公共图片
        image_url = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
        print(f"Downloading test image from {image_url}...")
        response = requests.get(image_url, timeout=10)
        response.raise_for_status()
        
        print(f"Image downloaded. Size: {len(response.content)} bytes")
        image = Image.open(BytesIO(response.content))
        print(f"Image opened successfully: {image.format}, {image.size}, {image.mode}")

        # 构建对话消息
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_url},
                    {"type": "text", "text": "详细描述这张图片的内容。"},
                ],
            }
        ]

        # 生成符合模型要求的文本 Prompt
        prompt_text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        print(f"Formatted Prompt: {prompt_text}")
        
        # 4. 构建 vLLM 输入
        # vLLM 接收 prompt 文本和多模态数据字典
        inputs = {
            "prompt": prompt_text,
            "multi_modal_data": {
                "image": image
            },
        }

        # 5. 设置采样参数
        sampling_params = SamplingParams(temperature=0.1, max_tokens=512)

        # 6. 执行推理
        print("Running generation...")
        outputs = llm.generate([inputs], sampling_params=sampling_params)

        for output in outputs:
            generated_text = output.outputs[0].text
            print("\n" + "="*20 + " Generated Output " + "="*20)
            print(generated_text)
            print("="*58 + "\n")
            
    except Exception as e:
        print(f"Error during inference: {e}")

if __name__ == "__main__":
    run_inference()
