#!/bin/bash
# MAE 消融實驗腳本
# 用於重現論文 Table 1a 和 Table 1b 的結果

echo "========================================================================"
echo "MAE 消融實驗 - 解碼器配置對 Fine-tuning 的影響"
echo "========================================================================"
echo ""

# # 清理舊的模型和日誌
rm -rf models/ablation
rm -rf logs/cifar10/ablation

# 創建保存目錄
mkdir -p models/ablation
mkdir -p logs/cifar10/ablation

# ============================================================================
echo "========================================================================"
echo "實驗 (a): 解碼器深度"
echo "配置: decoder_dim=192 (固定), decoder_layer=[2,8] (變化)" # default decoder_layer=4, decoder_dim=192
echo "========================================================================"
echo ""

# 16
for layers in 2 8
do
    echo "--------------------------------------------------------------------"
    echo "實驗 a-${layers}: decoder_layer=${layers}, decoder_dim=192"
    echo "--------------------------------------------------------------------"
    
    # 階段 1: MAE 預訓練
    echo "[階段 1/2] MAE 預訓練..."
    python mae_pretrain.py \
        --decoder_layer ${layers} \
        --model_path models/ablation/mae_depth_${layers}-final.pt # 必須以"-final.pt"結尾，才能正確生成 best 模型
    
    if [ $? -ne 0 ]; then
        echo "❌ MAE 預訓練失敗"
        continue
    fi
    
    # 階段 2: Fine-tuning 分類器
    echo "[階段 2/2] Fine-tuning 分類器..."
    python train_classifier.py \
        --pretrained_model_path models/ablation/mae_depth_${layers}-best.pt \
        --output_model_path models/ablation/classifier_depth_${layers}.pt
    
    if [ $? -eq 0 ]; then
        echo "✅ 實驗 a-${layers} 完成"
    else
        echo "❌ Fine-tuning 失敗"
    fi
    echo ""
done


# ============================================================================

echo ""
echo "========================================================================"
echo "實驗 (b): 解碼器寬度"
echo "配置: decoder_layer=4 (固定), decoder_dim=[96, 288, 384] (變化)" # default decoder_layer=4, decoder_dim=192
echo "========================================================================"
echo ""

# 288
for dim in 96 384
do
    echo "--------------------------------------------------------------------"
    echo "實驗 b-${dim}: decoder_layer=4, decoder_dim=${dim}"
    echo "--------------------------------------------------------------------"
    
    # 階段 1: MAE 預訓練
    echo "[階段 1/2] MAE 預訓練..."
    python mae_pretrain.py \
        --decoder_layer 4 \
        --decoder_dim ${dim} \
        --total_epoch 200 \
        --warmup_epoch 20 \
        --model_path models/ablation/mae_width_${dim}-final.pt
    
    if [ $? -ne 0 ]; then
        echo "❌ MAE 預訓練失敗"
        continue
    fi
    
    # 階段 2: Fine-tuning 分類器
    echo "[階段 2/2] Fine-tuning 分類器..."
    python train_classifier.py \
        --pretrained_model_path models/ablation/mae_width_${dim}-best.pt \
        --output_model_path models/ablation/classifier_width_${dim}.pt \
        --total_epoch 100 \
        --warmup_epoch 5
    
    if [ $? -eq 0 ]; then
        echo "✅ 實驗 b-${dim} 完成"
    else
        echo "❌ Fine-tuning 失敗"
    fi
    echo ""
done

echo ""
echo "========================================================================"
echo "🎉 所有消融實驗完成!"
echo "========================================================================"
echo ""
echo "結果位置:"
echo "  - 模型: models/ablation/"
echo "  - 日誌: logs/cifar10/"
echo ""
echo "查看訓練曲線:"
echo "  tensorboard --logdir logs/cifar10"
echo "========================================================================"













echo ""
echo "========================================================================"
echo "生成分類器可視化結果 (用於論文展示)"
echo "========================================================================"
echo ""

# 創建可視化輸出目錄
mkdir -p pic/ablation

# 可視化所有消融實驗的分類器結果
echo "--------------------------------------------------------------------"
echo "可視化消融實驗 (a): 解碼器深度"
echo "--------------------------------------------------------------------"

for layers in 2 8
do
    if [ -f "models/ablation/classifier_depth_${layers}.pt" ]; then
        echo "正在可視化 classifier_depth_${layers}..."
        python visualize_classifier.py \
            --model_path models/ablation/classifier_depth_${layers}.pt \
            --output_dir pic/ablation \
            --num_samples 16 \
            --scale_factor 4
        
        if [ $? -eq 0 ]; then
            echo "✅ classifier_depth_${layers} 可視化完成"
        else
            echo "❌ classifier_depth_${layers} 可視化失敗"
        fi
    else
        echo "⚠️  模型文件不存在: models/ablation/classifier_depth_${layers}.pt"
    fi
    echo ""
done

echo ""
echo "--------------------------------------------------------------------"
echo "可視化消融實驗 (b): 解碼器寬度"
echo "--------------------------------------------------------------------"

for dim in 96 384
do
    if [ -f "models/ablation/classifier_width_${dim}.pt" ]; then
        echo "正在可視化 classifier_width_${dim}..."
        python visualize_classifier.py \
            --model_path models/ablation/classifier_width_${dim}.pt \
            --output_dir pic/ablation \
            --num_samples 16 \
            --scale_factor 4
        
        if [ $? -eq 0 ]; then
            echo "✅ classifier_width_${dim} 可視化完成"
        else
            echo "❌ classifier_width_${dim} 可視化失敗"
        fi
    else
        echo "⚠️  模型文件不存在: models/ablation/classifier_width_${dim}.pt"
    fi
    echo ""
done

echo ""
echo "--------------------------------------------------------------------"
echo "可視化基準模型"
echo "--------------------------------------------------------------------"

# 可視化預訓練模型的分類器
if [ -f "vit-t-classifier-from_pretrained.pt" ]; then
    echo "正在可視化 pretrained 分類器..."
    python visualize_classifier.py \
        --model_path vit-t-classifier-from_pretrained.pt \
        --output_dir pic \
        --num_samples 16 \
        --scale_factor 4
    
    if [ $? -eq 0 ]; then
        echo "✅ pretrained 分類器可視化完成"
    else
        echo "❌ pretrained 分類器可視化失敗"
    fi
else
    echo "⚠️  模型文件不存在: vit-t-classifier-from_pretrained.pt"
fi
echo ""

# 可視化從頭訓練的分類器
if [ -f "vit-t-classifier-from_scratch.pt" ]; then
    echo "正在可視化 from-scratch 分類器..."
    python visualize_classifier.py \
        --model_path vit-t-classifier-from_scratch.pt \
        --output_dir pic \
        --num_samples 16 \
        --scale_factor 4
    
    if [ $? -eq 0 ]; then
        echo "✅ from-scratch 分類器可視化完成"
    else
        echo "❌ from-scratch 分類器可視化失敗"
    fi
else
    echo "⚠️  模型文件不存在: vit-t-classifier-from_scratch.pt"
fi
echo ""

echo ""
echo "========================================================================"
echo "🎉 所有可視化完成!"
echo "========================================================================"
echo ""
echo "可視化結果位置:"
echo "  - 消融實驗: pic/ablation/"
echo "  - 基準模型: pic/"
echo ""
echo "每個模型生成的文件包括:"
echo "  1. *_predictions.png - 預測結果可視化 (適合放入論文)"
echo "  2. *_confusion_matrix.png - 混淆矩陣"
echo "  3. *_results.txt - 詳細評估結果"
echo "========================================================================"




