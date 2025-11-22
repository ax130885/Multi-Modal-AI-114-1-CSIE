
# Multi-Modal AI: LLaVA LoRA Fine-tuning & Zero-Shot Inference

## Environment Setup

```bash
conda create -n swift python=3.10
conda activate swift
pip install ms-swift transformers datasets torch pillow matplotlib pandas nltk msgspec
```

## LLaVA Zero-Shot Inference

1. Place your images in the `llava_vqa_finetune/data/zero_shot` directory.
2. Run the notebook: `task1_zero_shot.ipynb`.
3. The results will be saved in `llava_vqa_finetune/outputs/zero_shot`.

## LLaVA LoRA Fine-tuning

Run the notebook: `task2_lora_finetune.ipynb`.

Notebook outline:
1. Download VQA-RAD Dataset
2. Data Preprocessing
3. Fine-Tuning
4. Merge LoRA Weights
5. Evaluation (BLEU) (using `eval.py`)
6. Qualitative Comparison (Visualization)

**Tip:** For steps 3, 4, and 5, it is recommended to open a separate terminal, navigate to `llava_vqa_finetune`, and execute the scripts there instead of running them directly in the notebook. This helps prevent CUDA OOM (out-of-memory) errors due to VRAM not being released properly.




