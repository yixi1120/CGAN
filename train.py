import os
# 环境变量设置 - 解决那个烦人的OpenMP警告
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn.functional as F
import numpy as np
from scipy.stats import entropy
from torch.utils.tensorboard import SummaryWriter
import datetime

from simple_cnn_model import SimpleCNNEdge2Image
from utils import EdgeImageDataset, save_checkpoint, save_samples


def calculate_kl_divergence(p, q):
    """计算KL散度 - 衡量两个分布的差异"""
    # 先转成概率分布
    p = F.softmax(p.view(p.size(0), -1), dim=1).cpu().numpy()
    q = F.softmax(q.view(q.size(0), -1), dim=1).cpu().numpy()
    
    # 计算KL散度平均值
    kl_vals = [entropy(p[i], q[i]) for i in range(p.shape[0])]
    return np.mean(kl_vals)


def calculate_fcn_score(real_imgs, fake_imgs, fcn_model):
    """计算FCN分数"""
    with torch.no_grad():
        # 提取特征
        real_feats = fcn_model(real_imgs)['out']
        fake_feats = fcn_model(fake_imgs)['out']
        
        # 计算特征空间中的L1距离
        fcn_score = F.l1_loss(fake_feats, real_feats)
    return fcn_score.item()


def evaluate_model(model, val_loader, device, fcn_model=None):
    """评估模型性能"""
    model.eval()
    total_kl = 0.0
    total_fcn = 0.0
    batch_count = 0
    
    with torch.no_grad():
        for edge, real in val_loader:
            # 数据准备
            edge = edge.to(device)
            real = real.to(device)
            
            # 生成图像
            fake = model(edge)
            
            # 计算KL散度
            kl_val = calculate_kl_divergence(real, fake)
            total_kl += kl_val
            
            # 如果有FCN模型，计算FCN分数
            if fcn_model is not None:
                fcn_val = calculate_fcn_score(real, fake, fcn_model)
                total_fcn += fcn_val
            
            batch_count += 1
    
    # 计算平均分数
    avg_kl = total_kl / batch_count
    metrics = {'kl_divergence': avg_kl}
    
    if fcn_model is not None:
        avg_fcn = total_fcn / batch_count
        metrics['fcn_score'] = avg_fcn
    
    return metrics


def make_grid_image(tensor, normalize=False):
    """处理图像用于TensorBoard显示"""
    # CPU上处理
    tensor = tensor.cpu().detach()
    
    # 归一化到0-1范围
    if normalize:
        tensor = (tensor * 0.5 + 0.5).clamp(0, 1)
    
    # 只要第一张
    return tensor[0]


def train(args):
    """训练边缘图到真实图像的CNN模型"""
    # 基本设置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 创建TensorBoard记录器
    log_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    log_dir = os.path.join(args.log_dir, f"train_{log_time}")
    tb_writer = SummaryWriter(log_dir)
    print(f"TensorBoard日志: {log_dir}")

    # 创建目录
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.sample_dir, exist_ok=True)

    # 数据预处理
    img_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # 归一化到[-1,1]
    ])

    # 加载训练数据
    train_dataset = EdgeImageDataset(
        root_dir=args.data_dir,
        mode='train',
        transform=img_transform
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )

    # 加载验证数据（如果有）
    val_dataset = None
    val_loader = None
    if args.val_dir:
        val_dataset = EdgeImageDataset(
            root_dir=args.val_dir,
            mode='val',
            transform=img_transform
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers
        )

    # 创建模型
    model = SimpleCNNEdge2Image(
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        ngf=args.ngf
    ).to(device)
    
    # 模型信息
    print(model)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters())}")

    # 选择损失函数 - L1(MAE)或MSE
    criterion = nn.L1Loss() if args.loss == 'l1' else nn.MSELoss()
    print(f"损失函数: {args.loss.upper()}")
    
    # 设置优化器 - Adam
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.5, 0.999))
    print(f"优化器: Adam (lr={args.lr})")
    
    # 学习率调度器
    scheduler = None
    if args.lr_decay:
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, 
            step_size=args.lr_decay_epochs, 
            gamma=args.lr_decay_rate
        )
        print(f"使用学习率调度: 每{args.lr_decay_epochs}轮衰减为{args.lr_decay_rate}倍")

    # 加载预训练FCN（用于评估）
    fcn_model = None
    if args.fcn_model_path and os.path.exists(args.fcn_model_path):
        print(f"加载FCN模型: {args.fcn_model_path}")
        fcn_model = torch.hub.load('pytorch/vision:v0.10.0', 'fcn_resnet50', pretrained=True)
        fcn_model = fcn_model.to(device)
        fcn_model.eval()

    # 开始训练
    print(f"\n{'='*20} 开始训练 {'='*20}")
    print("使用简单CNN方法 - 直接用重建损失训练，无判别器")
    
    step_count = 0
    best_loss = float('inf')
    
    for epoch in range(args.epochs):
        # 训练模式
        model.train()
        epoch_start = time.time()
        epoch_loss = 0.0
        
        for batch_idx, (edge, real) in enumerate(train_loader):
            # 数据准备
            edge = edge.to(device)
            real = real.to(device)
            
            # 生成图像
            fake = model(edge)
            
            # 计算损失
            loss = criterion(fake, real)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 更新统计
            step_count += 1
            epoch_loss += loss.item()
            
            # 记录到TensorBoard
            tb_writer.add_scalar('损失/训练', loss.item(), step_count)
            
            # 记录学习率
            cur_lr = scheduler.get_last_lr()[0] if scheduler else args.lr
            tb_writer.add_scalar('学习率', cur_lr, step_count)
            
            # 打印进度
            if batch_idx % args.print_freq == 0:
                print(f"轮次 [{epoch+1}/{args.epochs}] | "
                      f"批次 [{batch_idx}/{len(train_loader)}] | "
                      f"损失: {loss.item():.4f}")
            
            # 保存生成样本
            if step_count % args.sample_freq == 0:
                # 保存图像
                sample_path = os.path.join(args.sample_dir, f"sample_{step_count}.png")
                save_samples(edge, real, fake, sample_path)
                print(f"样本已保存: {sample_path}")
                
                # TensorBoard可视化
                tb_writer.add_image('Edges', make_grid_image(edge, normalize=True), step_count)
                tb_writer.add_image('Real', make_grid_image(real, normalize=True), step_count)
                tb_writer.add_image('Generated', make_grid_image(fake, normalize=True), step_count)
        
        # 更新学习率
        if scheduler:
            scheduler.step()
        
        # 计算平均损失
        avg_epoch_loss = epoch_loss / len(train_loader)
        epoch_time = time.time() - epoch_start
        
        print(f"轮次 [{epoch+1}/{args.epochs}] 完成 | "
              f"平均损失: {avg_epoch_loss:.4f} | "
              f"用时: {epoch_time:.2f}秒")
        
        # 记录轮次损失
        tb_writer.add_scalar('损失/轮次', avg_epoch_loss, epoch)
        
        # 验证评估
        if val_loader:
            print("执行验证...")
            metrics = evaluate_model(model, val_loader, device, fcn_model)
            
            kl_val = metrics['kl_divergence']
            print(f"KL散度: {kl_val:.4f}")
            
            if 'fcn_score' in metrics:
                fcn_val = metrics['fcn_score']
                print(f"FCN分数: {fcn_val:.4f}")
            
            # 记录验证指标
            tb_writer.add_scalar('验证/KL散度', kl_val, epoch)
            if 'fcn_score' in metrics:
                tb_writer.add_scalar('验证/FCN分数', fcn_val, epoch)
            
            # 保存最佳模型
            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss
                save_checkpoint(
                    model, 
                    optimizer, 
                    epoch, 
                    os.path.join(args.checkpoint_dir, "best_model.pth")
                )
                print("保存最佳模型！")
        
        # 定期保存检查点
        if (epoch + 1) % args.save_freq == 0:
            ckpt_path = os.path.join(args.checkpoint_dir, f"checkpoint_{epoch+1}.pth")
            save_checkpoint(model, optimizer, epoch, ckpt_path)
            print(f"检查点已保存: {ckpt_path}")
    
    # 保存最终模型
    final_path = os.path.join(args.checkpoint_dir, "final_model.pth")
    save_checkpoint(model, optimizer, args.epochs, final_path)
    print(f"最终模型已保存: {final_path}")
    
    # 添加模型图到TensorBoard
    dummy_input = torch.randn(1, args.in_channels, args.image_size, args.image_size).to(device)
    tb_writer.add_graph(model, dummy_input)
    
    # 关闭TensorBoard
    tb_writer.close()
    
    print(f"训练完成！使用以下命令查看训练过程：\ntensorboard --logdir={args.log_dir}")
    print("\n模型总结:")
    print("- 简单的边缘图到图像转换CNN")
    print("- 直接使用重建损失训练，无对抗损失")
    print("- 训练时间更短，实现更简单，但可能细节表现不如cGAN")


if __name__ == "__main__":
    # 命令行参数
    parser = argparse.ArgumentParser(description="CNN边缘图生成器训练脚本")
    
    # 数据参数
    parser.add_argument("--data_dir", 
                        type=str, 
                        default='processed/train', 
                        help="训练数据目录")
    parser.add_argument("--val_dir", type=str, default="", 
                        help="验证数据目录")
    parser.add_argument("--image_size", type=int, default=256, 
                        help="图像大小")
    parser.add_argument("--in_channels", type=int, default=1, 
                        help="输入通道数(边缘图)")
    parser.add_argument("--out_channels", type=int, default=3, 
                        help="输出通道数(RGB)")
    
    # 模型参数
    parser.add_argument("--ngf", type=int, default=64, 
                        help="特征图数量")
    
    # 训练参数
    parser.add_argument("--batch_size", type=int, default=4, 
                        help="批次大小")
    parser.add_argument("--epochs", type=int, default=100, 
                        help="训练轮数")
    parser.add_argument("--lr", type=float, default=0.0001, 
                        help="学习率")
    parser.add_argument("--loss", type=str, default="l1", 
                        choices=["l1", "mse"], 
                        help="损失函数(l1=绝对误差,mse=均方误差)")
    parser.add_argument("--num_workers", type=int, default=4, 
                        help="数据加载线程数")
    
    # 学习率调度参数
    parser.add_argument("--lr_decay", action="store_true", 
                        help="是否使用学习率衰减")
    parser.add_argument("--lr_decay_epochs", type=int, default=50, 
                        help="学习率衰减步长")
    parser.add_argument("--lr_decay_rate", type=float, default=0.5, 
                        help="学习率衰减率")
    
    # 输出参数
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", 
                        help="模型保存目录")
    parser.add_argument("--sample_dir", type=str, default="samples", 
                        help="样本保存目录")
    parser.add_argument("--save_freq", type=int, default=10, 
                        help="保存检查点频率(轮数)")
    parser.add_argument("--sample_freq", type=int, default=500, 
                        help="保存样本频率(步数)")
    parser.add_argument("--print_freq", type=int, default=100, 
                        help="打印信息频率(批次)")
    
    # 评估参数
    parser.add_argument("--fcn_model_path", type=str, default="", 
                        help="FCN模型路径(用于计算FCN分数)")
    
    # TensorBoard参数
    parser.add_argument("--log_dir", type=str, default="runs", 
                        help="TensorBoard日志目录")
    
    # 解析参数
    args = parser.parse_args()
    
    # 创建日志目录
    os.makedirs(args.log_dir, exist_ok=True)
    
    # 开始训练
    train(args) 