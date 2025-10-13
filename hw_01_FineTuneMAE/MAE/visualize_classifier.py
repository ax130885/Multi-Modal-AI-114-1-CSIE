#!/usr/bin/env python3
"""
分類器結果可視化腳本
用於可視化已訓練的分類器在測試集上的預測結果,無需重新訓練
"""
import os
import argparse
import torch
import torchvision
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from torchvision.transforms import ToTensor, Compose, Normalize
from tqdm import tqdm

from model import ViT_Classifier
from utils import setup_seed

# CIFAR-10 類別名稱
CIFAR10_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']

def visualize_predictions(model, dataset, device, num_samples=16, output_path='classifier_visualization.png', scale_factor=4):
    """
    可視化分類器的預測結果
    
    Args:
        model: 訓練好的分類器模型
        dataset: 測試數據集
        device: 計算設備
        num_samples: 要可視化的樣本數量 (建議 16, 25, 36 等完全平方數)
        output_path: 輸出圖片路徑
        scale_factor: 圖片放大倍數
    """
    model.eval()
    
    # 計算網格大小 (盡量接近正方形)
    grid_size = int(np.sqrt(num_samples))
    actual_samples = grid_size * grid_size
    
    print(f'正在可視化 {actual_samples} 個樣本 ({grid_size}x{grid_size} 網格)...')
    
    with torch.no_grad():
        # 取出指定數量的圖片
        images = torch.stack([dataset[i][0] for i in range(actual_samples)])
        labels = torch.tensor([dataset[i][1] for i in range(actual_samples)])
        
        images = images.to(device)
        labels = labels.to(device)
        
        # 預測
        logits = model(images)
        predictions = logits.argmax(dim=-1)
        
        # 計算準確率
        accuracy = (predictions == labels).float().mean().item()
        print(f'這 {actual_samples} 個樣本的準確率: {accuracy*100:.2f}%')
        
        # 將圖片放大以便查看
        images_large = F.interpolate(images.cpu(), scale_factor=scale_factor, mode='nearest')
        
        # 將圖片從 [-1, 1] 轉換到 [0, 1]
        images_large = (images_large + 1) / 2
        
        # 創建可視化
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(grid_size*2, grid_size*2))
        fig.suptitle(f'Classifier Predictions (Accuracy: {accuracy*100:.2f}%)', fontsize=16, y=0.995)
        
        for idx in range(actual_samples):
            row = idx // grid_size
            col = idx % grid_size
            ax = axes[row, col] if grid_size > 1 else axes
            
            # 顯示圖片
            img = images_large[idx].permute(1, 2, 0).numpy()
            img = np.clip(img, 0, 1)  # 確保在有效範圍內
            ax.imshow(img)
            
            # 設置標題 (真實標籤 vs 預測標籤)
            true_label = CIFAR10_CLASSES[labels[idx].item()]
            pred_label = CIFAR10_CLASSES[predictions[idx].item()]
            is_correct = predictions[idx] == labels[idx]
            
            title_color = 'green' if is_correct else 'red'
            title = f'GT: {true_label}\nPred: {pred_label}'
            ax.set_title(title, fontsize=10, color=title_color)
            ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f'✅ 可視化結果已保存到: {output_path}')
        plt.close()

def evaluate_model(model, dataloader, device):
    """
    在整個數據集上評估模型性能
    
    Args:
        model: 訓練好的分類器模型
        dataloader: 數據加載器
        device: 計算設備
    
    Returns:
        accuracy: 準確率
        per_class_acc: 每個類別的準確率
    """
    model.eval()
    
    all_predictions = []
    all_labels = []
    
    print('正在評估模型...')
    with torch.no_grad():
        for img, label in tqdm(dataloader):
            img = img.to(device)
            label = label.to(device)
            
            logits = model(img)
            predictions = logits.argmax(dim=-1)
            
            all_predictions.append(predictions.cpu())
            all_labels.append(label.cpu())
    
    all_predictions = torch.cat(all_predictions)
    all_labels = torch.cat(all_labels)
    
    # 總體準確率
    accuracy = (all_predictions == all_labels).float().mean().item()
    
    # 每個類別的準確率
    per_class_acc = {}
    for cls_idx, cls_name in enumerate(CIFAR10_CLASSES):
        mask = all_labels == cls_idx
        if mask.sum() > 0:
            cls_acc = (all_predictions[mask] == all_labels[mask]).float().mean().item()
            per_class_acc[cls_name] = cls_acc
    
    return accuracy, per_class_acc

def create_confusion_matrix_plot(model, dataloader, device, output_path='confusion_matrix.png'):
    """
    創建並保存混淆矩陣圖
    
    Args:
        model: 訓練好的分類器模型
        dataloader: 數據加載器
        device: 計算設備
        output_path: 輸出圖片路徑
    """
    model.eval()
    
    # 初始化混淆矩陣
    num_classes = len(CIFAR10_CLASSES)
    confusion_matrix = torch.zeros(num_classes, num_classes, dtype=torch.int64)
    
    print('正在生成混淆矩陣...')
    with torch.no_grad():
        for img, label in tqdm(dataloader):
            img = img.to(device)
            label = label.to(device)
            
            logits = model(img)
            predictions = logits.argmax(dim=-1)
            
            # 更新混淆矩陣
            for true_label, pred_label in zip(label.cpu(), predictions.cpu()):
                confusion_matrix[true_label, pred_label] += 1
    
    # 繪製混淆矩陣
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # 將混淆矩陣轉換為百分比
    confusion_matrix_pct = confusion_matrix.float() / confusion_matrix.sum(dim=1, keepdim=True) * 100
    
    im = ax.imshow(confusion_matrix_pct.numpy(), cmap='Blues', aspect='auto')
    
    # 設置刻度和標籤
    ax.set_xticks(np.arange(num_classes))
    ax.set_yticks(np.arange(num_classes))
    ax.set_xticklabels(CIFAR10_CLASSES, rotation=45, ha='right')
    ax.set_yticklabels(CIFAR10_CLASSES)
    
    # 在每個格子中顯示數值
    for i in range(num_classes):
        for j in range(num_classes):
            text = ax.text(j, i, f'{confusion_matrix[i, j]}\n({confusion_matrix_pct[i, j]:.1f}%)',
                          ha="center", va="center", color="black" if confusion_matrix_pct[i, j] < 50 else "white",
                          fontsize=8)
    
    ax.set_title('Confusion Matrix', fontsize=16, pad=20)
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    
    # 添加顏色條
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Percentage (%)', rotation=270, labelpad=20)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f'✅ 混淆矩陣已保存到: {output_path}')
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='可視化分類器預測結果')
    parser.add_argument('--model_path', type=str, required=True, help='訓練好的分類器模型路徑')
    parser.add_argument('--output_dir', type=str, default='pic', help='輸出目錄')
    parser.add_argument('--num_samples', type=int, default=16, help='可視化的樣本數量 (建議使用完全平方數)')
    parser.add_argument('--batch_size', type=int, default=512, help='評估時的 batch size')
    parser.add_argument('--seed', type=int, default=42, help='隨機種子')
    parser.add_argument('--scale_factor', type=int, default=4, help='圖片放大倍數')
    parser.add_argument('--skip_confusion_matrix', action='store_true', help='跳過混淆矩陣生成')
    
    args = parser.parse_args()
    
    setup_seed(args.seed)
    
    # 創建輸出目錄
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 提取模型名稱作為文件前綴
    model_name = os.path.splitext(os.path.basename(args.model_path))[0]
    
    # 載入數據集
    print('載入 CIFAR-10 測試集...')
    test_dataset = torchvision.datasets.CIFAR10('data', train=False, download=True, 
                                                 transform=Compose([ToTensor(), Normalize(0.5, 0.5)]))
    test_dataloader = torch.utils.data.DataLoader(test_dataset, args.batch_size, 
                                                   shuffle=False, num_workers=4)
    
    # 載入模型
    print(f'載入模型: {args.model_path}')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = torch.load(args.model_path, map_location=device)
    model.to(device)
    model.eval()
    
    # 1. 評估整體性能
    print('\n' + '='*70)
    print('評估模型性能')
    print('='*70)
    accuracy, per_class_acc = evaluate_model(model, test_dataloader, device)
    
    print(f'\n總體準確率: {accuracy*100:.2f}%')
    print('\n每個類別的準確率:')
    print('-'*40)
    for cls_name, cls_acc in per_class_acc.items():
        print(f'{cls_name:12s}: {cls_acc*100:.2f}%')
    
    # 保存評估結果到文本文件
    results_path = os.path.join(args.output_dir, f'{model_name}_results.txt')
    with open(results_path, 'w') as f:
        f.write(f'Model: {args.model_path}\n')
        f.write(f'Overall Accuracy: {accuracy*100:.2f}%\n\n')
        f.write('Per-class Accuracy:\n')
        f.write('-'*40 + '\n')
        for cls_name, cls_acc in per_class_acc.items():
            f.write(f'{cls_name:12s}: {cls_acc*100:.2f}%\n')
    print(f'\n✅ 評估結果已保存到: {results_path}')
    
    # 2. 可視化預測結果
    print('\n' + '='*70)
    print('生成預測可視化')
    print('='*70)
    vis_path = os.path.join(args.output_dir, f'{model_name}_predictions.png')
    visualize_predictions(model, test_dataset, device, 
                         num_samples=args.num_samples, 
                         output_path=vis_path,
                         scale_factor=args.scale_factor)
    
    # 3. 生成混淆矩陣
    if not args.skip_confusion_matrix:
        print('\n' + '='*70)
        print('生成混淆矩陣')
        print('='*70)
        cm_path = os.path.join(args.output_dir, f'{model_name}_confusion_matrix.png')
        create_confusion_matrix_plot(model, test_dataloader, device, output_path=cm_path)
    
    print('\n' + '='*70)
    print('🎉 所有可視化完成!')
    print('='*70)
    print(f'\n生成的文件:')
    print(f'  - 預測可視化: {vis_path}')
    if not args.skip_confusion_matrix:
        print(f'  - 混淆矩陣: {cm_path}')
    print(f'  - 評估結果: {results_path}')
    print('='*70)
