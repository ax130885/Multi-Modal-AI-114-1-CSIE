import os
import argparse
import math
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import ToTensor, Compose, Normalize
from tqdm import tqdm
import torch.nn.functional as F

from model import *
from utils import setup_seed

def calculate_psnr(img1, img2, max_val=1.0):
    """Calculate PSNR between two images"""
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    psnr = 20 * torch.log10(max_val / torch.sqrt(mse))
    return psnr.item()

def calculate_ssim(img1, img2, window_size=11, size_average=True):
    """Calculate SSIM between two images (simplified version)"""
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    mu1 = F.avg_pool2d(img1, window_size, 1, padding=window_size//2)
    mu2 = F.avg_pool2d(img2, window_size, 1, padding=window_size//2)
    
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = F.avg_pool2d(img1 * img1, window_size, 1, padding=window_size//2) - mu1_sq
    sigma2_sq = F.avg_pool2d(img2 * img2, window_size, 1, padding=window_size//2) - mu2_sq
    sigma12 = F.avg_pool2d(img1 * img2, window_size, 1, padding=window_size//2) - mu1_mu2
    
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    if size_average:
        return ssim_map.mean().item()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=4096) # 梯度累積後的等效 batch size
    parser.add_argument('--max_device_batch_size', type=int, default=512) # 每個 GPU 的最大 batch size
    parser.add_argument('--base_learning_rate', type=float, default=1.5e-4)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--mask_ratio', type=float, default=0.75)
    # parser.add_argument('--total_epoch', type=int, default=2000)
    # parser.add_argument('--warmup_epoch', type=int, default=200)
    parser.add_argument('--total_epoch', type=int, default=200)
    parser.add_argument('--warmup_epoch', type=int, default=20)
    parser.add_argument('--model_path', type=str, default='vit-t-mae-final.pt')
    
    # 消融實驗參數: 用於測試不同的解碼器配置
    parser.add_argument('--decoder_layer', type=int, default=4, help='解碼器層數 (實驗 a: 1,2,4,8,12)')
    parser.add_argument('--decoder_dim', type=int, default=None, help='解碼器維度 (實驗 b: 128,256,512,768,1024), 若為 None 則與 encoder 相同')
    parser.add_argument('--log_dir', type=str, default=None, help='TensorBoard 日誌目錄 (若為 None 則自動生成)')

    args = parser.parse_args()

    setup_seed(args.seed)

    batch_size = args.batch_size
    load_batch_size = min(args.max_device_batch_size, batch_size)

    assert batch_size % load_batch_size == 0
    steps_per_update = batch_size // load_batch_size

    # transform: totensor 資料會從 [0, 255] 變成 [0, 1], normalize 後會變成 [-1, 1]
    train_dataset = torchvision.datasets.CIFAR10('data', train=True, download=True, transform=Compose([ToTensor(), Normalize(0.5, 0.5)]))
    val_dataset = torchvision.datasets.CIFAR10('data', train=False, download=True, transform=Compose([ToTensor(), Normalize(0.5, 0.5)]))
    dataloader = torch.utils.data.DataLoader(train_dataset, load_batch_size, shuffle=True, num_workers=4)
    
    # 根據實驗配置生成獨立的日誌目錄,避免不同實驗的日誌互相覆蓋
    if args.log_dir is None:
        # 如果使用默認參數,使用默認日誌目錄
        if args.decoder_layer == 4 and args.decoder_dim is None:
            log_dir = os.path.join('logs', 'cifar10', 'mae-pretrain')
        else:
            # 消融實驗:根據配置生成唯一的日誌目錄名稱
            decoder_dim_str = str(args.decoder_dim) if args.decoder_dim is not None else '192'
            log_dir = os.path.join('logs', 'cifar10', 'ablation', f'mae_layer{args.decoder_layer}_dim{decoder_dim_str}')
    else:
        log_dir = args.log_dir
    
    writer = SummaryWriter(log_dir)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 創建 MAE 模型,支援自定義解碼器配置
    # encoder 維度固定為 192, encoder 層數固定為 12
    # 可調整的是: decoder_layer (解碼器層數) 和 decoder_emb_dim (解碼器維度)
    model = MAE_ViT(
        image_size=32,
        patch_size=2,
        emb_dim=192,           # encoder 維度 (固定)
        encoder_layer=12,      # encoder 層數 (固定)
        encoder_head=3,
        decoder_layer=args.decoder_layer,      # 解碼器層數 (可調整,用於實驗 a)
        decoder_head=3,
        mask_ratio=args.mask_ratio,
        decoder_emb_dim=args.decoder_dim       # 解碼器維度 (可調整,用於實驗 b)
    ).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.base_learning_rate * args.batch_size / 256, betas=(0.9, 0.95), weight_decay=args.weight_decay)
    lr_func = lambda epoch: min((epoch + 1) / (args.warmup_epoch + 1e-8), 0.5 * (math.cos(epoch / args.total_epoch * math.pi) + 1))
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_func, verbose=True)

    step_count = 0
    optim.zero_grad()
    best_ssim = 0.0  # 記錄最佳 SSIM 值
    
    for e in range(args.total_epoch):
        model.train()
        losses = []
        for img, label in tqdm(iter(dataloader)):
            step_count += 1
            img = img.to(device)
            predicted_img, mask = model(img)
            loss = torch.mean((predicted_img - img) ** 2 * mask) / args.mask_ratio
            loss.backward()
            if step_count % steps_per_update == 0: # 梯度累積 每 8 epoch 更新一次
                optim.step()
                optim.zero_grad()
            losses.append(loss.item())
        lr_scheduler.step()
        avg_loss = sum(losses) / len(losses)
        writer.add_scalar('tr/lr', optim.param_groups[0]['lr'], global_step=e)
        writer.add_scalar('tr/MSE_loss', avg_loss, global_step=e)
        print(f'In epoch {e}, average traning loss is {avg_loss}.')

        ''' visualize the first 16 predicted images on val dataset and compute metrics'''
        model.eval()
        with torch.no_grad():
            val_img = torch.stack([val_dataset[i][0] for i in range(16)]) # for 從驗證集取前16張圖 # stack 將圖片拼成一個 batch [16, c, h, w]
            val_img = val_img.to(device)
            predicted_val_img, mask = model(val_img) # predicted_val_img 修復的完整圖片 # mask 在 enc 當中被隨機遮蔽的位置

            # # 原始寫法
            # predicted_val_img = predicted_val_img * mask + val_img * (1 - mask)
            # img = torch.cat([val_img * (1 - mask), predicted_val_img, val_img], dim=0)
            
            ############# 修改段落 增加評估指標 保存 best model 以及線性插值放大可視化結果4x4倍 (原一張圖只有32x32看到眼睛脫窗) #############
            # 計算評估指標（只在被遮罩的區域）
            masked_region = predicted_val_img * mask # pred
            original_masked = val_img * mask         # gt
            
            # 將圖片範圍從 [-1, 1] 轉換到 [0, 1] 以便計算指標 (從 transform 的 Normalize(0.5, 0.5) 可知)
            masked_region_norm = (masked_region + 1) / 2
            original_masked_norm = (original_masked + 1) / 2
            
            # MSE (on masked region only)
            mse = torch.mean((masked_region - original_masked) ** 2).item()
            
            # MAE (on masked region only)
            mae = torch.mean(torch.abs(masked_region - original_masked)).item()
            
            # PSNR (on masked region only)
            # Peak Signal-to-Noise Ratio: 峰值信噪比，通常在 20-40 dB 之間，越高越好
            psnr = calculate_psnr(masked_region_norm, original_masked_norm, max_val=1.0)
            
            # SSIM (on full reconstructed image) 
            # Structural Similarity Index: 結構相似性，範圍 0-1，越接近 1 表示重建品質越好
            predicted_val_img_full = predicted_val_img * mask + val_img * (1 - mask) # pred的mask區域 + gt的非mask區域
            predicted_val_img_full_norm = (predicted_val_img_full + 1) / 2           # 從 [-1, 1] norm到 [0, 1]
            val_img_norm = (val_img + 1) / 2                                         # 從 [-1, 1] norm到 [0, 1]
            ssim = calculate_ssim(predicted_val_img_full_norm, val_img_norm)
            
            # 記錄指標到 TensorBoard
            writer.add_scalar('val/MSE_metrics', mse, global_step=e)
            writer.add_scalar('val/MAE_metrics', mae, global_step=e)
            writer.add_scalar('val/PSNR_metrics', psnr, global_step=e)
            writer.add_scalar('val/SSIM_metrics', ssim, global_step=e)
            
            print(f'Validation Metrics - MSE: {mse:.6f}, MAE: {mae:.6f}, PSNR: {psnr:.2f} dB, SSIM: {ssim:.4f}')
            
            # 如果當前 SSIM 是最佳的，保存 best 模型
            if ssim > best_ssim:
                best_ssim = ssim
                best_model_path = args.model_path.replace('final.pt', 'best.pt')
                torch.save(model, best_model_path)
                print(f'🎉 New best model saved with SSIM: {ssim:.4f} at epoch {e}')
            


            # # 紀錄可視化圖片到 tensorborad            
            # 使用 F.interpolate 將圖片放大 4x4 倍（從 32x32 到 128x128）
            val_img_large = F.interpolate(val_img, scale_factor=4, mode='nearest')
            masked_img_large = F.interpolate(val_img * (1 - mask), scale_factor=4, mode='nearest')
            predicted_img_large = F.interpolate(predicted_val_img_full, scale_factor=4, mode='nearest')
            
            # # 將 batch 內的所有結果可視化後拚成大圖
            # [16*3, c, h, w] # 16為batch size, 3是每個batch三種圖 (mask | pred | gt )
            img = torch.cat([masked_img_large, predicted_img_large, val_img_large], dim=0)

            # # 大圖 寬2*3張子圖 高8張子圖
            # 把 16(batch)*3(種圖) 拆成 (v=3, h1, w1=2) (可以算出h1=8)
            # 大圖的高=h1(數量)*h(單個圖片高度), 寬=w1(數量)*v(三種圖)*w(單個圖片寬度)
            # img = rearrange(img, '(v h1 w1) c h w -> c (h1 h) (w1 v w)', w1=2, v=3) 

            # # 大圖 寬16張子圖 高3種圖
            # 結果會是 mask1 && mask2 && ... \\ pred1 && pred2 && ... \\ gt1 && gt2 && ...
            img = rearrange(img, '(v n) c h w -> c (v h) (n w)', v=3)  # 3行16列
            writer.add_image('val/visualize', (img + 1) / 2, global_step=e)
        
        ''' save model '''
        torch.save(model, args.model_path)