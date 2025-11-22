# ms-swift 程式碼庫結構描述

## 概述
ms-swift 是一個用於模型微調的開源框架，專注於 LoRA (Low-Rank Adaptation) 和其他參數高效微調技術。該框架支援多種模型架構，並提供簡化的微調流程。

## 主要目錄結構

### 根目錄檔案
- **README.md / README_CN.md**: 專案說明文件，提供快速入門指南
- **setup.py / setup.cfg**: 設置和打包配置
- **requirements.txt**: 項目依賴包列表
- **Makefile**: 編譯和開發工具命令
- **LICENSE / CODE_OF_CONDUCT.md / CONTRIBUTING.md**: 授權和貢獻指南
- **.gitignore, .pre-commit-config.yaml**: 開發配置文件

### 核心源碼目錄 (swift/)
- **cli/**: 命令行介面功能
  - `main.py`: CLI 主入口,透過 `setup.py` 註冊為 `swift` 命令
  - `sft.py`: 訓練命令入口,調用 `swift.llm.sft_main()`
  - `export.py`: 導出命令入口,調用 `swift.llm.export_main()`
  - 其他: `eval.py`, `infer.py`, `deploy.py`, `merge_lora.py` 等
  
- **llm/**: 大語言模型相關實現 (核心模組)
  - **train/**: 訓練相關實現
    - `sft.py`: SFT 訓練主邏輯 (`SwiftSft` 類,`sft_main` 函數)
    - `tuner.py`: 微調器應用邏輯 (`TunerMixin` 類)
    - `pt.py`: 預訓練相關
    - `rlhf.py`: RLHF 訓練相關
  - **export/**: 導出相關實現
    - `export.py`: 導出主邏輯 (`SwiftExport` 類,`export_main` 函數)
    - `merge_lora.py`: LoRA 權重合併實現
    - `quant.py`: 模型量化實現 (AWQ, GPTQ, BNB)
    - `ollama.py`: 導出至 Ollama 格式
  - **model/**: 模型註冊與載入
    - `register.py`: 模型註冊機制
    - `model_arch.py`: 模型架構定義 (如 `llava_hf`)
    - `model/llava.py`: LLaVA 模型專用載入器
    - `utils.py`: 模型工具函數 (`get_model_tokenizer`)
  - **template/**: 提示模板系統
    - `base.py`: 模板基類
    - `template/`: 各模型的提示模板實現
  - **dataset/**: 數據集處理
    - `loader.py`: 數據載入器
    - 預處理器: `AlpacaPreprocessor`, `MessagesPreprocessor`
  - **argument/**: 參數定義
    - `train_args.py`: 訓練參數 (`TrainArguments`)
    - `export_args.py`: 導出參數 (`ExportArguments`)
    - `tuner_args.py`: 微調參數 (`TunerArguments`)

- **tuners/**: 微調器模組實現
  - `lora.py`: LoRA 實現 (`LoRA` 類, `LoRAConfig`)
  - `lora_layers.py`: LoRA 層實現
  - `adapter.py`: Adapter 微調
  - `peft.py`: PEFT 整合
  - `base.py`: 微調器基類
  
- **trainers/**: 訓練器實現
  - 繼承自 Hugging Face Transformers 的 `Trainer`
  - 實現各種訓練策略 (DDP, DeepSpeed 等)
  
- **hub/**: 模型中心相關功能,用於模型的上傳和下載
- **megatron/**: 支援 Megatron 框架的整合
- **plugin/**: 外掛系統,擴展功能模組
- **ray/**: 支援 Ray 分散式計算框架
- **ui/**: 使用者介面組件
- **utils/**: 工具函數和輔助功能
- **__init__.py, version.py**: 初始化和版本信息

### 範例目錄 (examples/)
- **app/**: 應用程式範例
- **custom/**: 自定義實現範例
- **deploy/**: 部署相關範例
- **eval/**: 評估腳本範例
- **export/**: 模型導出範例
- **infer/**: 推理腳本範例
- **megatron/**: Megatron 相關範例
- **models/**: 不同模型的使用範例
- **notebook/**: Jupyter Notebook 範例
- **sampler/**: 採樣器範例
- **train/**: 訓練腳本範例
- **yaml/**: 配置文件範例

### 其他重要目錄
- **docs/**: 詳細的文檔和 API 參考
- **tests/**: 測試套件和單元測試
- **scripts/**: 各種工具腳本
- **requirements/**: 按功能分類的依賴管理
- **asset/**: 資源文件和資產
- **.github/**: GitHub 工作流和模板
- **.dev_scripts/**: 開發用腳本

## 功能特點
1. **多模型支援**: 支援各種大語言模型的微調
2. **參數高效微調**: 實現 LoRA、QLoRA 等技術，大幅減少微調所需的參數
3. **分散式訓練**: 支援多 GPU 和多節點訓練
4. **易於使用**: 提供命令行工具和簡單的 API
5. **靈活配置**: 通過 YAML 和配置文件支援靈活的實驗配置

---

## 命令執行流程詳解

### 1. `swift sft` 命令執行流程

#### 命令範例
```bash
swift sft \
    --model llava-hf/llava-1.5-7b-hf \
    --model_type llava1_5_hf \
    --train_type lora \
    --dataset ./data/train.jsonl \
    --output_dir ./outputs/llava_lora \
    --use_hf true \
    --torch_dtype bfloat16 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-4 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --target_modules all-linear \
    --gradient_checkpointing true
```

#### 執行流程

1. **命令入口** (`swift/cli/sft.py`)
   ```python
   from swift.llm import sft_main
   sft_main()
   ```
   - 透過 `setup.py` 註冊的 `swift` 命令調用
   - Entry point: `swift=swift.cli.main:cli_main`

2. **主邏輯** (`swift/llm/train/sft.py`)
   - 創建 `SwiftSft` 類實例
   - 執行流程:
     ```
     SwiftSft.__init__()
       ├── _prepare_model_tokenizer()    # 載入模型和 tokenizer
       ├── _prepare_template()            # 設置提示模板
       ├── _prepare_callbacks()           # 設置訓練回調
       └── _prepare_flash_ckpt()          # Flash checkpoint 支援
     
     SwiftSft.run()
       ├── _prepare_dataset()             # 準備訓練數據
       │   ├── _get_dataset()            # 載入原始數據
       │   ├── _encode_dataset()         # 數據編碼
       │   └── _post_process_datasets()  # 後處理 (packing 等)
       ├── prepare_model()                # 應用 LoRA
       │   └── TunerMixin.prepare_model()
       │       └── Swift.prepare_model() # 添加 LoRA 層
       ├── TrainerFactory.get_trainer_cls() # 獲取訓練器
       └── trainer.train()                # 開始訓練
     ```

3. **關鍵組件**

   **a. 參數解析** (`swift/llm/argument/train_args.py`)
   - `TrainArguments`: 訓練主參數
   - `TunerArguments`: LoRA 相關參數
     - `--lora_rank`: LoRA 矩陣秩 (預設 8)
     - `--lora_alpha`: LoRA 縮放因子 (預設 32)
     - `--lora_dropout`: Dropout 率
     - `--target_modules`: 要應用 LoRA 的模組 (如 `all-linear`)
   - `Seq2SeqTrainingArguments`: 繼承自 HF Transformers

   **b. 模型載入** (`swift/llm/model/`)
   - `model/llava.py`: LLaVA 模型註冊
     ```python
     register_model(ModelMeta(
         MLLMModelType.llava1_5_hf,
         model_groups=[...],
         template=TemplateType.llava1_5_hf,
         get_function=get_model_tokenizer_llava_hf,
         architectures=['LlavaForConditionalGeneration'],
         model_arch=ModelArch.llava_hf
     ))
     ```
   - `utils.py`: `get_model_tokenizer()` 函數
     - 從 Hugging Face Hub 下載模型
     - 載入 `LlavaForConditionalGeneration`
     - 載入 `AutoProcessor` (包含 tokenizer 和 image processor)

   **c. LoRA 應用** (`swift/tuners/lora.py`)
   - `LoRAConfig`: LoRA 配置類
     ```python
     @dataclass
     class LoRAConfig(LoraConfig, SwiftConfig):
         r: int = 8              # lora_rank
         lora_alpha: int = 32
         lora_dropout: float = 0.05
         target_modules: List[str]
     ```
   - `LoRA.prepare_model()`: 將 LoRA 層添加到模型
     - 使用 PEFT 的 `LoraModel` 包裝原始模型
     - 只訓練 LoRA 參數,凍結原始權重
   
   **d. 數據處理** (`swift/llm/dataset/`)
   - `load_dataset()`: 載入 JSONL 格式數據
   - `Template.encode()`: 編碼為模型輸入格式
     - 處理 `<image>` 標記
     - 生成 `input_ids`, `attention_mask`, `labels`
   - `LazyLLMDataset`: 延遲編碼 (節省記憶體)

   **e. 訓練器** (`swift/trainers/`)
   - 繼承自 HF `Seq2SeqTrainer`
   - 支援 DeepSpeed, DDP 等分散式訓練
   - 自動保存 checkpoint 到 `output_dir`

4. **輸出結果**
   - `output_dir/` 結構:
     ```
     ./outputs/llava_lora/
     ├── v0-20251120-171300/           # 時間戳版本目錄
     │   ├── checkpoint-339/           # 訓練 checkpoint
     │   │   ├── adapter_config.json  # LoRA 配置
     │   │   ├── adapter_model.bin    # LoRA 權重
     │   │   ├── trainer_state.json   # 訓練狀態
     │   │   └── ...
     │   ├── configuration.json        # 模型配置
     │   └── args.json                # 訓練參數
     ```

---

### 2. `swift export` 命令執行流程

#### 命令範例
```bash
swift export \
    --adapters ./outputs/llava_lora/v0-20251120/checkpoint-339 \
    --use_hf true \
    --merge_lora true
```

#### 執行流程

1. **命令入口** (`swift/cli/export.py`)
   ```python
   from swift.llm import export_main
   export_main()
   ```

2. **主邏輯** (`swift/llm/export/export.py`)
   - 創建 `SwiftExport` 類實例
   - 執行流程:
     ```
     SwiftExport.run()
       ├── if args.to_peft_format:      # (可選) 轉換為 PEFT 格式
       │     swift_to_peft_format()
       ├── if args.merge_lora:          # ✓ 合併 LoRA 權重
       │     merge_lora()
       ├── if args.quant_method:        # (可選) 量化模型
       │     quantize_model()
       ├── if args.to_ollama:           # (可選) 導出至 Ollama
       │     export_to_ollama()
       └── if args.push_to_hub:         # (可選) 上傳至 Hub
             hub.push_to_hub()
     ```

3. **LoRA 合併詳解** (`swift/llm/export/merge_lora.py`)

   **a. 函數: `merge_lora(args: ExportArguments)`**
   ```python
   def merge_lora(args, device_map=None, replace_if_exists=False):
       # 1. 設置輸出目錄
       output_dir = args.output_dir or f'{args.adapters[0]}-merged'
       
       # 2. 載入模型和 LoRA 適配器
       args.quant_method = None  # 禁用量化
       model, template = prepare_model_template(args)
       
       # 3. 合併 LoRA 權重
       check_tie_word_embeddings(model)
       Swift.merge_and_unload(model)  # 核心操作
       model = model.model
       
       # 4. 保存合併後的模型
       save_checkpoint(
           model, template.processor, output_dir,
           safe_serialization=args.safe_serialization,
           max_shard_size=args.max_shard_size
       )
   ```

   **b. 關鍵步驟解析**

   - **載入模型**: `prepare_model_template(args)`
     1. 從 `adapters[0]` 讀取 `adapter_config.json`
     2. 識別 base model (如 `llava-hf/llava-1.5-7b-hf`)
     3. 載入 base model
     4. 使用 PEFT 載入 LoRA 適配器
        ```python
        from peft import PeftModel
        model = PeftModel.from_pretrained(
            base_model, 
            adapter_path,
            adapter_name="default"
        )
        ```

   - **合併權重**: `Swift.merge_and_unload(model)`
     - 對每個帶 LoRA 的層:
       ```
       W_merged = W_base + (LoRA_B @ LoRA_A) * (alpha / r)
       ```
     - 移除 LoRA 適配器層
     - 返回原始模型架構

   - **保存模型**: `save_checkpoint()`
     - 保存合併後的權重: `pytorch_model.bin` 或 `model.safetensors`
     - 保存配置: `config.json`
     - 保存 processor: `preprocessor_config.json`, `tokenizer_config.json`
     - 支援分片保存 (`max_shard_size`)

4. **參數說明** (`swift/llm/argument/export_args.py`)

   ```python
   @dataclass
   class ExportArguments(MergeArguments, BaseArguments):
       # 基本參數
       adapters: List[str]           # LoRA checkpoint 路徑
       output_dir: Optional[str]     # 輸出目錄 (預設: {adapters[0]}-merged)
       
       # 合併相關
       merge_lora: bool = False      # 是否合併 LoRA
       safe_serialization: bool      # 使用 safetensors 格式
       max_shard_size: str           # 分片大小 (如 "5GB")
       
       # 量化相關 (可選)
       quant_method: Literal['awq', 'gptq', 'bnb', 'fp8']
       quant_bits: int               # 量化位數 (4, 8)
       
       # 其他
       to_ollama: bool               # 導出至 Ollama
       push_to_hub: bool             # 上傳至 ModelScope Hub
       hub_model_id: str             # Hub 模型 ID
   ```

5. **輸出結果**
   - `checkpoint-339-merged/` 結構:
     ```
     ./outputs/llava_lora/v0-20251120/checkpoint-339-merged/
     ├── config.json                  # 模型配置
     ├── generation_config.json       # 生成配置
     ├── pytorch_model.bin            # 合併後的權重 (或 .safetensors)
     ├── preprocessor_config.json     # Processor 配置
     ├── tokenizer_config.json        # Tokenizer 配置
     ├── tokenizer.json               # Tokenizer 詞表
     ├── special_tokens_map.json      # 特殊 token 映射
     └── ...
     ```
   - 可直接用於推理:
     ```python
     from transformers import LlavaForConditionalGeneration
     model = LlavaForConditionalGeneration.from_pretrained(
         "./outputs/llava_lora/v0-20251120/checkpoint-339-merged"
     )
     ```

---

## 關鍵技術點

### 1. LoRA (Low-Rank Adaptation)

**原理**:
- 在預訓練模型的線性層旁添加低秩分解矩陣
- 公式: $h = W_0 x + \frac{\alpha}{r} B A x$
  - $W_0$: 凍結的原始權重
  - $A \in \mathbb{R}^{r \times d}$, $B \in \mathbb{R}^{d \times r}$: 可訓練的低秩矩陣
  - $r$: 秩 (rank),通常 $r \ll d$
  - $\alpha$: 縮放因子

**實現** (`swift/tuners/lora_layers.py`):
```python
class Linear(LoraLayer):
    def forward(self, x):
        result = F.linear(x, self.weight, self.bias)  # 原始層
        if self.merged:
            return result
        else:
            # LoRA 計算
            lora_output = (self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T)
            return result + lora_output * self.scaling
```

**優勢**:
- 參數效率: 只訓練 ~0.1% 的參數
- 記憶體節省: 不需要梯度反傳至完整模型
- 模組化: 可載入/卸載不同的 LoRA 適配器

### 2. QLoRA (Quantized LoRA)

- 使用 4-bit 量化載入 base model
- LoRA 參數使用 full precision (fp16/bf16)
- 透過 `bitsandbytes` 實現
- 命令參數:
  ```bash
  --quantization_bit 4
  --bnb_4bit_compute_dtype bfloat16
  ```

### 3. 多模態訓練 (LLaVA)

**架構**:
```
Image → Vision Encoder (CLIP) → Projector → LLM (Llama)
                                    ↓
                                LoRA 應用在此
```

**訓練策略** (`swift/llm/train/tuner.py`):
```python
def get_multimodal_target_regex(
    model,
    freeze_llm=False,      # 凍結語言模型
    freeze_vit=True,       # 凍結視覺編碼器
    freeze_aligner=True    # 凍結投影層
):
    # 預設只訓練 LLM 的 LoRA 參數
    # VIT 和 Aligner 保持凍結
```

### 4. Gradient Checkpointing

- 用於節省顯存
- 參數: `--gradient_checkpointing true`
- 原理: 在前向傳播時不保存中間激活值,反向傳播時重新計算
- Trade-off: 記憶體 ↓, 計算時間 ↑

### 5. 數據格式

**Swift 格式** (JSONL):
```json
{
  "messages": [
    {"role": "user", "content": "<image>What is in the image?"},
    {"role": "assistant", "content": "A cat sitting on a table."}
  ],
  "images": ["/path/to/image.jpg"]
}
```

**Template 處理**:
- `<image>` token → 插入視覺特徵
- 自動添加系統提示
- 生成訓練所需的 `input_ids`, `attention_mask`, `labels`

---

## 常用參數速查

### 訓練參數

| 參數 | 說明 | 預設值 | 建議值 |
|------|------|--------|--------|
| `--model` | 模型 ID 或路徑 | - | `llava-hf/llava-1.5-7b-hf` |
| `--model_type` | 模型類型 | 自動推斷 | `llava1_5_hf` |
| `--train_type` | 訓練類型 | `lora` | `lora` (高效), `full` (完整微調) |
| `--lora_rank` | LoRA 秩 | 8 | 8-32 (越大越強但越慢) |
| `--lora_alpha` | LoRA alpha | 32 | `lora_rank * 4` |
| `--lora_dropout` | LoRA dropout | 0.05 | 0.05-0.1 |
| `--target_modules` | LoRA 目標模組 | `DEFAULT` | `all-linear`, `q_proj,v_proj` |
| `--num_train_epochs` | 訓練輪數 | 3 | 3-5 |
| `--learning_rate` | 學習率 | 1e-4 | 1e-4 (LoRA), 1e-5 (full) |
| `--per_device_train_batch_size` | Batch size | 1 | 4-8 (視 GPU 記憶體) |
| `--gradient_accumulation_steps` | 梯度累積 | 1 | 4-8 (模擬大 batch) |
| `--gradient_checkpointing` | 梯度檢查點 | false | true (節省顯存) |
| `--torch_dtype` | 模型精度 | auto | `bfloat16`, `float16` |

### 導出參數

| 參數 | 說明 | 預設值 |
|------|------|--------|
| `--adapters` | LoRA checkpoint 路徑 | - |
| `--merge_lora` | 合併 LoRA 權重 | false |
| `--output_dir` | 輸出目錄 | `{adapters[0]}-merged` |
| `--quant_method` | 量化方法 | None |
| `--quant_bits` | 量化位數 | - |
| `--to_ollama` | 導出至 Ollama | false |
| `--push_to_hub` | 上傳至 Hub | false |

---

## 故障排除

### 1. CUDA Out of Memory

**解決方案**:
```bash
# 減少 batch size
--per_device_train_batch_size 2

# 增加梯度累積
--gradient_accumulation_steps 8

# 啟用梯度檢查點
--gradient_checkpointing true

# 使用更小的 LoRA rank
--lora_rank 4
```

### 2. LoRA 合併失敗

**常見原因**:
- Base model 路徑錯誤
- `adapter_config.json` 中的 `base_model_name_or_path` 不正確

**解決方案**:
```bash
# 手動指定 base model
swift export \
    --adapters ./checkpoint-339 \
    --model llava-hf/llava-1.5-7b-hf \
    --merge_lora true
```

### 3. 模型載入錯誤

**檢查**:
- 確認模型類型: `--model_type llava1_5_hf`
- 確認 transformers 版本: `pip install transformers>=4.36`
- 檢查模型文件完整性

---

## 參考資源

### 官方文檔
- GitHub: https://github.com/modelscope/swift
- 文檔: https://swift.readthedocs.io/
- 範例: `examples/train/`, `examples/export/`

### 相關論文
- LoRA: [Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- QLoRA: [Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)
- LLaVA: [Visual Instruction Tuning](https://arxiv.org/abs/2304.08485)