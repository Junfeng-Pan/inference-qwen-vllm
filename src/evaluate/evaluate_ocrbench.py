import os
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import pynvml
import json
from datasets import load_from_disk
from vllm import LLM, SamplingParams
from transformers import AutoProcessor
from tqdm import tqdm

# 显存监控列表
memory_logs = []
time_logs = []

def record_memory():
    """
    使用 pynvml 获取真实的 GPU 显存使用量 (Used Memory)，
    这包含了 vLLM 预分配的显存 (KV Cache) 以及 PyTorch 占用的显存。
    """
    try:
        # 使用第 0 号 GPU
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        # 转换为 MB
        mem = info.used / (1024 * 1024)
        memory_logs.append(mem)
    except Exception:
        # 如果获取失败，记录 0 或使用 torch fallback
        if torch.cuda.is_available():
            memory_logs.append(torch.cuda.memory_allocated() / (1024 * 1024))
        else:
            memory_logs.append(0)

def evaluate():
    # 初始化 NVML
    try:
        pynvml.nvmlInit()
    except Exception as e:
        print(f"Warning: Failed to initialize NVML, memory logging may be inaccurate. Error: {e}")

    # 1. 路径设置

    
    model_dir = "./Qwen3-VL-8B-Instruct-AWQ"
    data_dir = "./data/OCRBench_HF"
    output_dir = "./results/Qwen3-VL-8B-Instruct-AWQ-minified/"
    os.makedirs(output_dir, exist_ok=True)
    

    '''
    model_dir = "./Qwen/Qwen3-VL-4B-Instruct"
    data_dir = "./data/OCRBench_HF"
    output_dir = "./results/Qwen3-VL-4B-Instruct/"
    os.makedirs(output_dir, exist_ok=True)
    '''

    '''
    model_dir = "./Qwen/Qwen3-VL-2B-Instruct"
    data_dir = "./data/OCRBench_HF"
    output_dir = "./results/Qwen3-VL-2B-Instruct/"
    os.makedirs(output_dir, exist_ok=True)
    '''
    
    # 2. 加载数据集
    print(f"Loading dataset from {data_dir}...")
    if not os.path.exists(data_dir):
        print(f"Dataset path {data_dir} does not exist. Please run download_ocrbench_hf.py first.")
        return
        
    dataset = load_from_disk(data_dir)
    dataset = dataset.select(range(300))
    print(f"Dataset size: {len(dataset)}")

    # 3. 初始化 vLLM
    print("Initializing vLLM...")
    try:
        llm = LLM(
            model=model_dir,
            trust_remote_code=True,
            # 增大上下文长度以容纳高分辨率 OCR 图片 (tokens > 16k)
            max_model_len=8192, 
            limit_mm_per_prompt={"image": 1},
            enforce_eager=True,
            # 提高显存利用率，给 KV Cache 更多空间
            gpu_memory_utilization=0.35, 
            # 关闭 vLLM 引擎层面的统计日志，防止刷屏
            disable_log_stats=True, 
        )
        processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
    except Exception as e:
        print(f"Failed to initialize vLLM: {e}")
        return

    # 4. 准备推理参数
    sampling_params = SamplingParams(temperature=0.0, max_tokens=128) # OCR 任务通常温度为 0
    
    correct_count = 0
    total_count = 0
    
    # 5. 推理循环
    print("Starting inference...")
    start_global = time.time()
    
    # 逐个样本处理方便统计，若追求极致速度可增大 batch_size
    batch_size = 1 
    
    results = []

    # tqdm 只在这里显示总体进度
    for i in tqdm(range(0, len(dataset), batch_size)):
        batch_samples = dataset[i : i + batch_size]
        batch_inputs = []
        batch_answers = batch_samples['answer']
        
        # 记录推理前显存
        record_memory()
        t0 = time.time()

        # 构建输入
        for j in range(len(batch_samples['image'])):
            image = batch_samples['image'][j]
            question = batch_samples['question'][j]
            
            if not question:
                question = "Describe the text in the image."

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": question},
                    ],
                }
            ]
            
            prompt_text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            batch_inputs.append({
                "prompt": prompt_text,
                "multi_modal_data": {"image": image}
            })

        # 执行推理
        try:
            # use_tqdm=False 防止每次 generate 都产生一个进度条
            outputs = llm.generate(batch_inputs, sampling_params=sampling_params, use_tqdm=False) 
        except Exception as e:
            print(f"Error in batch {i}: {e}")
            continue
            
        t1 = time.time()
        # 记录耗时 (平均每个样本)
        avg_time = (t1 - t0) / len(batch_inputs)
        for _ in range(len(batch_inputs)):
            time_logs.append(avg_time)
        
        # 记录推理后显存
        record_memory()

        # 评估准确率
        for j, output in enumerate(outputs):
            pred = output.outputs[0].text.strip()
            gt = batch_answers[j]
            
            # 简单的评估逻辑
            is_correct = False
            if isinstance(gt, str):
                if gt.lower() in pred.lower() or pred.lower() in gt.lower():
                    is_correct = True
            elif isinstance(gt, list):
                for ans in gt:
                    if ans.lower() in pred.lower():
                        is_correct = True
                        break
            
            if is_correct:
                correct_count += 1
            total_count += 1
            
            results.append({
                "question": batch_samples['question'][j],
                "ground_truth": gt,
                "prediction": pred,
                "is_correct": is_correct
            })

    total_time = time.time() - start_global
    accuracy = correct_count / total_count if total_count > 0 else 0
    avg_speed = total_count / total_time
    avg_latency = np.mean(time_logs) if time_logs else 0
    peak_memory = max(memory_logs) if memory_logs else 0
    
    # 生成报告内容
    report_lines = [
        "="*30,
        f"EVALUATION REPORT: {model_dir}",
        f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "="*30,
        f"Total Samples: {total_count}",
        f"Accuracy: {accuracy:.2%}",
        f"Total Time: {total_time:.2f} s",
        f"Average Speed: {avg_speed:.2f} samples/s",
        f"Average Latency: {avg_latency:.4f} s/sample",
        f"Peak Memory (Used): {peak_memory:.2f} MB",
        "="*30,
        ""
    ]
    report_text = "\n".join(report_lines)
    
    # 打印到终端
    print("\n" + report_text)
    
    # 保存到文件
    model_name_safe = os.path.basename(model_dir.strip("/"))
    report_file = f"{output_dir}/report_{model_name_safe}.txt"
    with open(report_file, "w") as f:
        f.write(report_text)
    print(f"Report saved to {report_file}")

    # 6. 可视化
    plot_performance(output_dir)
    
    # 保存详细结果
    with open(f"{output_dir}/detailed_results.json", "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

def plot_performance(output_dir):
    plt.figure(figsize=(12, 5))
    
    # 显存图
    plt.subplot(1, 2, 1)
    plt.plot(memory_logs, label='Memory Used (MB)')
    plt.xlabel('Step (Batch * 2)')
    plt.ylabel('Memory (MB)')
    plt.title('GPU Memory Usage Over Time')
    plt.grid(True)
    plt.legend()
    
    # 耗时图
    plt.subplot(1, 2, 2)
    plt.hist(time_logs, bins=20, color='skyblue', edgecolor='black')
    plt.xlabel('Inference Time (s)')
    plt.ylabel('Frequency')
    plt.title('Inference Latency Distribution')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/performance_metrics.png")
    print(f"Performance plots saved to {output_dir}/performance_metrics.png")

if __name__ == "__main__":
    evaluate()