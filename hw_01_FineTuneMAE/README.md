# Environment

```bash
conda env create -f environmet.yml
conda activate mae
```

# Pre-train
```bash
# output: vit-t-mae-best.pt, vit-t-mae-final.pt
python mae_pretrain.py

# loss & metrics shown in tensorboard
tensorboard --logdir logs/cifar10
```


# Fine-tune
## From Pre-train

```bash
# output: vit-t-classifier-from_pretrained.pt
python train_classifier.py --pretrained_model_path vit-t-mae-best.pt --output_model_path vit-t-classifier-from_pretrained.pt
```

## From Scratch

```bash
# output: vit-t-classifier-from_scratch.pt
python train_classifier.py
```

# Ablation Study & Confusion Matrix & Visualization
Ablation Study A: Try different decoder depth. Default=4, try=[2, 8].  
Ablation Study b: Try different decoder dimension. Default=192, try=[96, 384].  
```bash
# ablation study weight will be saved in 'models/'
# confusion matrix & visual results will be saved in `pic/`
bash run_ablation.sh
```

# Reference Repo
https://github.com/IcarusWizard/MAE?tab=readme-ov-file
