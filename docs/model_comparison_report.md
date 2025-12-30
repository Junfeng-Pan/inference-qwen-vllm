# Qwen3-VL 模型性能对比分析报告

## 1. 概述
本报告对比了 Qwen3-VL 系列三个不同规格模型在 OCRBench 任务上的表现。对比维度的包括：推理准确率、推理速度（延迟）以及显存占用。

**参测模型**：
1. **Qwen3-VL-8B-Instruct-AWQ**: 8B 参数量的 4-bit 量化版本 (AWQ)。
2. **Qwen3-VL-4B-Instruct**: 4B 参数量的原始精度版本 (BF16/FP16)。
3. **Qwen3-VL-2B-Instruct**: 2B 参数量的原始精度版本 (BF16/FP16)。

**测试环境**：
- **数据集**: OCRBench_HF (1000 样本)
- **推理框架**: vLLM 0.13.0
- **硬件**: NVIDIA GPU (测试时显存峰值约 19GB，总共显存32GB)

**一些参数**:

```python
llm = LLM(
            model=model_dir,
            trust_remote_code=True,
            # 增大上下文长度以容纳高分辨率 OCR 图片 (tokens > 16k)
            max_model_len=16400, 
            limit_mm_per_prompt={"image": 1},
            enforce_eager=True,
            # 限制显存利用率，给 KV Cache 更多空间
            gpu_memory_utilization=0.40, 
            # 关闭 vLLM 引擎层面的统计日志，防止刷屏
            disable_log_stats=True, 
        )
```
注：
* orcbench数据集中样本产生的最大上下文长度为16400左右，所以设置max_model_len=16400。
* **4B模型**由于显存不够用，所以设置gpu_memory_utilization=0.50。


## 2. 详细测试结果

| 指标 | Qwen3-VL-8B-AWQ | Qwen3-VL-4B | Qwen3-VL-2B |
| :--- | :--- | :--- | :--- |
| **准确率 (Accuracy)** | 77.40% | 79.80% | 75.30% |
| **平均延迟 (Latency)** | 0.99 s/sample | 0.73 s/sample | **0.57 s/sample** |
| **吞吐量 (Speed)** | 1.00 samples/s | 1.35 samples/s | **1.73 samples/s** |
| **显存峰值 (Memory)** | 15641 MB | 19515 MB | 16039 MB |
| **量化状态** | INT4 | BF16 (Native) | BF16 (Native) |



## 3. 分析与结论

### 3.1 准确率表现
*   三个模型都差不多，在75-80之间。

### 3.2 速度与延迟
*   **小模型优势明显**：推理速度与参数量成反比。**2B 模型** 最快，延迟仅为 0.57秒；**4B 模型** 第二 (0.73秒)， **量化8B模型**相比要慢点(0.99s/sample)。
*   **量化模型的速度**：8B-AWQ 虽然体积小，但计算量（FLOPs）主要取决于参数量（8B）和层数。虽然 INT4 计算有加速，但反量化开销和本来就大的参数规模使其速度（0.99秒）慢于 4B 和 2B 模型。

### 3.3 显存占用（奇怪的点）
*   **奇怪的点**：8B-AWQ 的显存占用比 2B 非量化版本还要低。
    *   注：在这两个模型的评测中我设置参数gpu_memory_utilization=0.40，限制显存利用率为0.4，两个模型静态分配的显存应该都是32×0.4=12.8GB左右，但是运行时显存占用2B模型应该更高，导致了这种情况。



## 4.缩小显存
我把最大上下文长度改成max_model_len=8192后，显存利用率gpu_memory_utilization=0.35后，使用前300个样本（分辨率不高，不会出错），显存峰值就缩到11483.38 MB了，感觉可以通过减小分辨率来降低显存。
输出如下：
```
==============================
EVALUATION REPORT: ./Qwen3-VL-8B-Instruct-AWQ
Date: 2025-12-30 10:39:56
==============================
Total Samples: 300
Accuracy: 91.33%
Total Time: 30.01 s
Average Speed: 10.00 samples/s
Average Latency: 0.0981 s/sample
Peak Memory (Used): 11483.38 MB
==============================
```


## 5.相关问题
* **我已经把显存利用率设置为0.4了，按理来说显存不会超过32×0.4=12.8，为什么显存最大是16GB？**
* **gemini回答如下：**
这是一个非常敏锐的观察！你发现了 **vLLM 参数配置与操作系统实际显存占用** 之间的“隐形差额”。

简单来说：**`gpu_memory_utilization` 控制的是“静态预留”的显存（权重 + KV Cache），但它管不住“动态运行时”产生的显存（Activations）和底层开销。**

这里有约 **3.2 GB** 的“额外”显存（16GB - 12.8GB），它们主要去哪了？让我们像剥洋葱一样拆解这 **16GB**：

### 1. 显存账单拆解

| 显存成分 | 来源 | 大小 (估算) | 是否受 0.4 参数控制？ |
| --- | --- | --- | --- |
| **1. CUDA Context** | 只要使用 GPU 就会产生的基础开销 | **~0.8 - 1.0 GB** | ❌ 否 (硬性开销) |
| **2. 模型权重** | Qwen3-VL-2B bf16 | **4.3 GB** | ✅ 是 (包含在预算内) |
| **3. KV Cache** | vLLM 根据 0.4 预算倒推出来的缓存池 | **6.35 GB** | ✅ 是 (由预算决定) |
| **4. 视觉塔激活值 (Vision Activations)** | **罪魁祸首**。处理图片时产生的临时张量 | **~3.5 - 4.0 GB** | ❌ **否 (动态申请)** |
| **总计** |  | **~15 - 16 GB** |  |
