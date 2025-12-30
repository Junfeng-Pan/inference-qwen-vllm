# Qwen3-VL vLLM Inference & Evaluation

本项目提供了一套完整的工具链，用于下载、推理和评估 Qwen3-VL 系列模型（包括 8B AWQ 量化版、4B 和 2B 原生版本）。基于 vLLM 推理框架，实现了高效的多模态内容生成与 OCR 能力评测。

## 📁 目录结构

```
.
├── Qwen3-VL-8B-Instruct-AWQ/  # (下载后) 8B AWQ 量化模型
├── Qwen/                      # (下载后) 4B/2B 对比模型
├── data/                      # 评测数据集 (OCRBench)
├── docs/                      # 文档与报告
├── results/                   # 评测结果与图表
├── src/
│   ├── download/              # 模型与数据下载脚本
│   ├── evaluate/              # 性能评测脚本 (OCRBench)
│   └── scripts/               # 单次推理演示脚本
├── requirements.txt           # 项目依赖
└── README.md                  # 本文件
```

## 🚀 快速开始

### 1. 环境准备

推荐使用 Linux 环境及 NVIDIA 显卡（显存 16GB+ 推荐）。

```bash
# 安装依赖
pip install -r requirements.txt
```

*注意：vLLM 对 PyTorch 版本有特定要求，安装过程中可能会自动调整你的 torch 版本。*

### 2. 模型下载

**下载 8B AWQ 量化模型** (主模型):
```bash
python src/download/download_model.py
```

**下载 4B & 2B 对比模型** (可选):
```bash
python src/download/download_comparison_models.py
```

### 3. 数据集准备

下载 OCRBench 数据集用于评测：
```bash
python src/download/download_ocrbench_hf.py
```

### 4. 运行推理演示

使用 8B AWQ 模型对一张网络图片进行描述：
```bash
python src/scripts/inference_vllm.py
```

### 5. 性能评测

运行 OCRBench 评测脚本，评估准确率、推理速度和显存占用：

```bash
# 默认评测 8B AWQ 模型
python src/evaluate/evaluate_ocrbench.py
```

*若需评测其他模型，请修改 `src/evaluate/evaluate_ocrbench.py` 中的 `model_dir` 变量。*

评测结果将保存在 `results/` 目录下，包含详细 JSON 数据、性能图表及文本报告。

## 📊 性能对比

我们在 OCRBench 上对比了三个模型的表现，详细分析请见 [docs/model_comparison_report.md](docs/model_comparison_report.md)。

**简要结论**：
*   **准确率**：Qwen3-VL-4B (79.8%) > 8B-AWQ (77.4%) > 2B (75.3%)
*   **显存占用**：8B-AWQ 因量化优势，显存占用最低 (~15.3GB)，甚至低于 2B 模型。
*   **推荐**：显存受限 (16GB) 场景推荐 **8B-AWQ**；显存充裕场景推荐 **4B**。

## 🛠️ 常见问题

1.  **推理报错 `decoder prompt is too long`**
    *   这是由于 OCR 图片分辨率高导致 token 数过多。我们在评测脚本中已将 `max_model_len` 调大至 16400 以解决此问题。

## 📄 License

本项目代码遵循 MIT 协议。引用的模型和数据集遵循其各自的开源协议。
