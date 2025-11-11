# hw_02_LoRA_CLIP

## Project Overview

This project explores different methods for adapting the pre-trained CLIP (Contrastive Language-Image Pre-training) model for image classification tasks on specific datasets. It focuses on two main approaches:
1.  **Zero-Shot Evaluation:** Assessing CLIP's out-of-the-box classification capabilities using various text prompt engineering strategies.
2.  **CLIP Fine-Tuning:** Adapting the visual encoder of CLIP using parameter-efficient fine-tuning techniques, specifically Linear Probing and LoRA (Low-Rank Adaptation), on Flowers102 and CUB-200 datasets.

The goal is to understand CLIP's performance in zero-shot settings and to compare the effectiveness and efficiency of Linear Probing and LoRA for domain-specific adaptation.

## Folder Structure

```
/mnt/data1/graduate/yuxin/class/Multi-Modal-AI-114-1-CSIE/hw_02_LoRA_CLIP/
├───cat_to_name.json                # Mapping for Flowers102 class IDs to names
├───Task1.ipynb                     # Jupyter Notebook for Zero-Shot Evaluation
├───Task2.ipynb                     # Jupyter Notebook for CLIP Fine-Tuning (Linear Probing & LoRA)
├───requirements.txt                # Python dependencies for the project
├───data/                           # Datasets (Flowers102, CUB-200 will be downloaded here)
│   └───flowers-102/
│       ├───102flowers.tgz
│       ├───imagelabels.mat
│       ├───setid.mat
│       └───jpg/...
├───my_photo/                       # Directory for custom test images
│   ├───bird.jpg
│   ├───bird.png
│   └───flower.png
```

## Environment Setup

To set up the development environment, follow these steps:

1.  **Create and Activate Conda Environment:**
    ```bash
    conda create -n lora_clip python=3.10
    conda activate lora_clip
    pip install -r requirements.txt
    ```

## Task 1: Zero-Shot Evaluation

`Task1.ipynb` focuses on evaluating the zero-shot capabilities of the pre-trained CLIP model on Flowers102 and CUB-200 datasets.

**Content:**
*   **CLIP Model Loading:** Loads the `openai/clip-vit-large-patch14` model and its processor.
*   **Dataset Preparation:** Prepares Flowers102 and CUB-200 datasets for testing. The `cat_to_name.json` file is used to map Flowers102 class IDs to human-readable names.
*   **Prompt Engineering:** Explores different prompt templates (default, diverse, domain-specific) to generate text embeddings for zero-shot classification.
*   **Evaluation:** Evaluates the model's accuracy on both datasets using the generated text prompts.
*   **Results & Visualization:** Compares the performance of different prompt templates, visualizes confusion matrices, and displays sample predictions.
*   **Custom Image Testing:** Includes a section to test custom images (e.g., `my_photo/flower.png`, `my_photo/bird.png`) and predict their classes using the zero-shot approach.

**Key Findings:**
*   The choice of prompt template significantly impacts zero-shot accuracy. Domain-specific prompts generally yield better results.
*   CLIP demonstrates strong zero-shot generalization, even on unseen categories.

## Task 2: CLIP Fine-Tuning (Linear Probing & LoRA)

`Task2.ipynb` explores fine-tuning the visual encoder of CLIP using two parameter-efficient methods: Linear Probing and LoRA.

**Content:**
*   **Dataset Preparation:** Flowers102 and CUB-200 datasets are split into training, validation, and test sets.
*   **Linear Probing:**
    *   The CLIP visual encoder backbone (`vision_model` and `visual_projection`) is frozen.
    *   Only a small, task-specific `LinearClassifier` head is trained on top of the frozen features.
    *   Training curves (loss and accuracy) are plotted, and the best model is saved (`flowers102_linear_probe_best.pth`, `cub200_linear_probe_best.pth`).
    *   Test set evaluation includes accuracy, classification reports, and confusion matrices.
*   **LoRA (Low-Rank Adaptation) Fine-Tuning:**
    *   Memory is cleared from previous Linear Probing models.
    *   A fresh CLIP visual encoder is loaded.
    *   LoRA is configured with parameters like `r=8`, `lora_alpha=16`, targeting `q_proj` and `v_proj` modules within the transformer layers.
    *   The LoRA adapters and a new `LinearClassifier` head are trained.
    *   Training curves are plotted, and the best LoRA adapter and classifier are saved (`flowers102_lora_best/`, `flowers102_lora_classifier_best.pth`, `cub200_lora_best/`, `cub200_lora_classifier_best.pth`).
    *   Test set evaluation includes accuracy, classification reports, and confusion matrices.
*   **Results Comparison and Analysis:**
    *   A detailed comparison table summarizes the test accuracy and trainable parameters for both Linear Probing and LoRA on both datasets.
    *   Visualizations compare the accuracy and training dynamics (loss and validation accuracy curves) of the two methods.
    *   Discussion on performance, parameter efficiency, and training characteristics of Linear Probing vs. LoRA.
*   **VRAM Usage Analysis:** A dedicated section measures and compares the GPU VRAM consumption for each method, demonstrating their memory efficiency.

**Key Findings:**
*   Both Linear Probing and LoRA are highly effective for adapting CLIP to new classification tasks with significantly fewer trainable parameters compared to full fine-tuning.
*   LoRA generally achieves slightly higher accuracy than Linear Probing, especially on more complex datasets like CUB-200, by allowing some fine-tuning of the backbone features.
*   LoRA uses slightly more VRAM and has more trainable parameters than Linear Probing, but both remain very memory-efficient.

## Usage

To run the notebooks:

1.  Ensure your conda environment is activated and dependencies are installed.
2.  Start a Jupyter Notebook server from the project root:
    ```bash
    jupyter notebook
    ```
3.  Open `Task1.ipynb` or `Task2.ipynb` in your browser and run the cells sequentially.

**Note:** For `Task2.ipynb`, it is recommended to restart the kernel and clear all outputs before running the LoRA section, especially if you encounter GPU memory issues, as indicated in the notebook's VRAM analysis section.
