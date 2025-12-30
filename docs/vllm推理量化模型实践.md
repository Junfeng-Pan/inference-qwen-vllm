# vLLM 推理 Qwen3-VL AWQ 量化模型实践

本文档记录了在 Linux 环境下，使用 vLLM 框架加载并推理自定义 AWQ 量化版 `Qwen3-VL-8B-Instruct` 模型的完整流程，包括环境搭建、模型获取、配置修正及踩坑解决方案。

## 1. 背景
我们此前已使用 `llm-compressor` 对 `Qwen3-VL-8B-Instruct` 进行了 AWQ (Activation-aware Weight Quantization) 量化，生成了 4-bit 权重的量化模型。现在的目标是使用 vLLM 这一高性能推理引擎来加载该模型进行多模态推理。

## 2. 环境配置

### 2.1 安装依赖
在已有基础环境（Python 3.10+, CUDA 12.1+）上，安装 vLLM 及相关工具：

```bash
pip install vllm modelscope qwen-vl-utils
```

**注意**：`vllm` 对 PyTorch 版本有特定要求（如 v0.13.0 依赖 torch 2.9.0）。安装过程中，`pip` 可能会卸载环境中已有的旧版 PyTorch（如 2.7.0+cu128）并安装兼容版本。这是正常现象，无需担心，但需留意对其他共存项目的影响。

## 3. 模型准备

### 3.1 下载私有模型
由于量化模型存储在 ModelScope 的私有仓库中，需要通过 SDK Token 进行下载。

**下载脚本** (`src/download/download_model.py`):
```python
from modelscope import snapshot_download
from modelscope.hub.api import HubApi

# 登录 ModelScope (需替换为真实 Token)
api = HubApi()
api.login("YOUR_MS_TOKEN")

# 下载模型到当前目录
model_dir = snapshot_download("JunfengPan/Qwen3-VL-8B-Instruct-AWQ", cache_dir='.')
```

### 3.2 目录调整
为了路径简洁，我们将下载后的模型移动到了项目根目录：
```bash
mv JunfengPan/Qwen3-VL-8B-Instruct-AWQ ./Qwen3-VL-8B-Instruct-AWQ
rmdir JunfengPan
```

## 4. 遇到的问题与解决方案

### 4.1 问题一：vLLM 初始化参数冲突
**现象**：
在使用 vLLM 加载模型时，显式指定了 `quantization="awq"`：
```python
llm = LLM(model=model_dir, quantization="awq", ...)
```
**报错**：
```text
ValueError: Quantization method specified in the model config (compressed-tensors) does not match the quantization method specified in the `quantization` argument (awq).
```
**原因**：
`llm-compressor` 导出的模型在 `config.json` 中标记的量化方法是 `compressed-tensors`（这是 vLLM 支持的一种通用压缩格式），而我们强制指定了 `awq`，导致冲突。
**解决**：
移除 `quantization="awq"` 参数，让 vLLM 自动读取 `config.json` 中的配置。

### 4.2 问题二：配置文件字段校验失败
**现象**：
移除参数后，vLLM 初始化再次失败：
```text
Failed to initialize vLLM: 2 validation errors for VllmConfig
scale_dtype
  Extra inputs are not permitted
zp_dtype
  Extra inputs are not permitted
```
**原因**：
`llm-compressor` 较新版本导出的 `config.json` 中，`quantization_config.weights` 包含了 `scale_dtype` 和 `zp_dtype` 字段（值为 `null`）。当前版本的 vLLM 在解析配置时认为这是不允许的额外字段。
**解决**：
手动修改模型目录下的 `config.json` 文件（修改前已备份为 `config.json.bak`）：
1. 打开 `Qwen3-VL-8B-Instruct-AWQ/config.json`。
2. 定位到 `quantization_config` -> `config_groups` -> `group_0` -> `weights`。
3. 删除 `"scale_dtype": null` 和 `"zp_dtype": null` 这两行。

### 4.3 问题三：测试图片下载失败
**现象**：
推理脚本运行时报错 `cannot identify image file`。
**原因**：
原脚本使用的测试图片 URL 可能不稳定或下载不完整。
**解决**：
更换为 Qwen-VL 官方更稳定的示例图片链接，并增加了 `requests` 的状态码检查。

## 5. 最终推理脚本

修正后的推理脚本 (`src/scripts/inference_vllm.py`) 核心逻辑如下：

```python
from vllm import LLM, SamplingParams
from transformers import AutoProcessor
# ... (imports)

# 1. 初始化 vLLM (自动识别量化配置)
llm = LLM(
    model="./Qwen3-VL-8B-Instruct-AWQ",
    trust_remote_code=True,
    max_model_len=4096,
    limit_mm_per_prompt={"image": 1},
    enforce_eager=True, # 启用 eager 模式以提高兼容性
)

# 2. 准备 Prompt
processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
messages = [
    {"role": "user", "content": [
        {"type": "image", "image": image_url},
        {"type": "text", "text": "详细描述这张图片的内容。"}
    ]}
]
prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# 3. 推理
outputs = llm.generate(
    [{"prompt": prompt_text, "multi_modal_data": {"image": image}}],
    sampling_params=SamplingParams(temperature=0.1, max_tokens=512)
)
```

## 6. 验证结果
执行脚本后，模型成功加载并对测试图片（海滩上的人与狗）生成了极其详尽、准确的中文描述，证明 AWQ 量化模型在大幅降低显存占用的同时，保持了优秀的多模态理解能力。
