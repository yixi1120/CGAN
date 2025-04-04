import os
# 环境变量设置 - 解决那个烦人的OpenMP警告
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import argparse
import torch
import torch.nn.functional as F
import numpy as np
from scipy.stats import entropy
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import datetime

from simple_cnn_model import SimpleCNNEdge2Image
from utils import EdgeImageDataset, save_samples


def calculate_kl_divergence(p, q):
    """计算KL散度 - 衡量两个分布的差异"""
    # 先转成概率分布
    p = F.softmax(p.view(p.size(0), -1), dim=1).cpu().numpy()
    q = F.softmax(q.view(q.size(0), -1), dim=1).cpu().numpy()
    
    # 计算KL散度平均值
    kl_vals = [entropy(p[i], q[i]) for i in range(p.shape[0])]
    return np.mean(kl_vals)


def calculate_fcn_score(real_imgs, fake_imgs, fcn_model):
    """计算FCN相似度分数（0-1范围）"""
    with torch.no_grad():
        # 提取特征
        real_feats = fcn_model(real_imgs)['out']
        fake_feats = fcn_model(fake_imgs)['out']
        
        # 计算L1距离
        l1_dist = F.l1_loss(fake_feats, real_feats)
        
        # 转换为相似度分数: 1=完全匹配, 0=完全不同
        sim_score = torch.exp(-l1_dist).item()
    
    return sim_score


def make_grid_image(tensor, normalize=False):
    """处理图像用于TensorBoard显示"""
    # CPU上处理
    tensor = tensor.cpu().detach()
    
    # 归一化到0-1范围
    if normalize:
        tensor = (tensor * 0.5 + 0.5).clamp(0, 1)
    
    # 只要第一张
    return tensor[0]


def test_model(args):
    """测试模型性能，计算质量指标"""
    # 基本设置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 创建TensorBoard记录器
    log_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    log_dir = os.path.join(args.log_dir, f"test_{log_time}")
    tb_writer = SummaryWriter(log_dir)
    print(f"TensorBoard日志: {log_dir}")

    # 数据预处理
    img_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # 加载测试数据
    test_dataset = EdgeImageDataset(
        root_dir=args.test_dir,
        mode='test',
        transform=img_transform
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,  # 使用参数中的批次大小
        shuffle=False,
        num_workers=args.num_workers
    )

    # 创建模型
    model = SimpleCNNEdge2Image(
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        ngf=args.ngf
    ).to(device)

    # 加载训练好的模型
    if os.path.exists(args.model_path):
        try:
            ckpt = torch.load(args.model_path, map_location=device, weights_only=False)
            # 如果是完整检查点
            if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
                model.load_state_dict(ckpt['model_state_dict'])
            else:
                # 如果只是状态字典
                model.load_state_dict(ckpt)
            print(f"模型加载成功: {args.model_path}")
        except Exception as e:
            print(f"模型加载失败: {e}")
            return
    else:
        print(f"模型文件不存在: {args.model_path}")
        return

    # 加载FCN模型
    print("加载FCN模型...")
    fcn_model = torch.hub.load('pytorch/vision:v0.10.0', 'fcn_resnet50', pretrained=True)
    fcn_model = fcn_model.to(device)
    fcn_model.eval()

    # 测试变量初始化
    model.eval()
    total_kl = 0.0
    total_fcn = 0.0
    fcn_scores = []
    kl_divs = []
    
    # 开始测试
    print("开始测试...")
    with torch.no_grad():
        for idx, (edge, real) in enumerate(tqdm(test_loader)):
            # 数据准备
            edge = edge.to(device)
            real = real.to(device)
            
            # 生成图像
            fake = model(edge)
            
            # 计算评估指标
            kl_val = calculate_kl_divergence(real, fake)
            fcn_val = calculate_fcn_score(real, fake, fcn_model)
            
            # 累计指标
            total_kl += kl_val
            total_fcn += fcn_val
            kl_divs.append(kl_val)
            fcn_scores.append(fcn_val)
            
            # 记录到TensorBoard
            tb_writer.add_scalar('Metrics/KL_Divergence', kl_val, idx)
            tb_writer.add_scalar('Metrics/FCN_Score', fcn_val, idx)
            
            # 保存部分样本和添加到TensorBoard
            if idx % args.sample_freq == 0:
                # TensorBoard可视化
                real_img = make_grid_image(real, normalize=True)
                fake_img = make_grid_image(fake, normalize=True)
                edge_img = make_grid_image(edge, normalize=True)
                
                tb_writer.add_image(f'Images/Real_{idx}', real_img, idx)
                tb_writer.add_image(f'Images/Generated_{idx}', fake_img, idx)
                tb_writer.add_image(f'Images/Edge_{idx}', edge_img, idx)
                
                # 创建并保存比较图
                def to_numpy(t):
                    return (t.cpu().detach() * 0.5 + 0.5).clamp(0, 1).permute(0, 2, 3, 1).numpy()
                
                # 转为numpy
                edge_np = to_numpy(edge)
                real_np = to_numpy(real)
                fake_np = to_numpy(fake)
                
                # 画图
                fig, ax = plt.subplots(1, 3, figsize=(12, 4))
                
                # 画边缘图
                if edge_np.shape[-1] == 1:
                    ax[0].imshow(edge_np[0, :, :, 0], cmap='gray')
                else:
                    ax[0].imshow(edge_np[0])
                ax[0].set_title('边缘图')
                ax[0].axis('off')
                
                # 画真实图和生成图
                ax[1].imshow(real_np[0])
                ax[1].set_title('真实图像')
                ax[1].axis('off')
                
                ax[2].imshow(fake_np[0])
                ax[2].set_title('生成图像')
                ax[2].axis('off')
                
                plt.tight_layout()
                save_path = os.path.join(args.output_dir, f"test_sample_{idx}.png")
                plt.savefig(save_path)
                plt.close()
                print(f"样本已保存: {save_path}")
    
    # 计算平均分数
    test_count = len(test_loader)
    avg_kl = total_kl / test_count
    avg_fcn = total_fcn / test_count
    
    # 打印结果
    print("\n测试结果:")
    print(f"平均KL散度: {avg_kl:.4f}")
    print(f"平均FCN分数: {avg_fcn:.4f} (范围0-1，越接近1越好)")
    
    # 记录平均分数到TensorBoard
    tb_writer.add_scalar('Metrics/Average_KL_Divergence', avg_kl, 0)
    tb_writer.add_scalar('Metrics/Average_FCN_Score', avg_fcn, 0)
    
    # 绘制分布图
    plt.figure(figsize=(12, 5))
    
    # KL散度分布
    plt.subplot(1, 2, 1)
    plt.hist(kl_divs, bins=min(20, len(kl_divs)), alpha=0.7, color='skyblue')
    plt.axvline(avg_kl, color='crimson', linestyle='dashed', linewidth=2)
    plt.title(f'KL散度分布 (平均: {avg_kl:.4f})')
    plt.xlabel('KL散度')
    plt.ylabel('频数')
    
    # FCN分数分布
    plt.subplot(1, 2, 2)
    plt.hist(fcn_scores, bins=min(20, len(fcn_scores)), alpha=0.7, color='lightgreen')
    plt.axvline(avg_fcn, color='crimson', linestyle='dashed', linewidth=2)
    plt.title(f'FCN分数分布 (平均: {avg_fcn:.4f})')
    plt.xlabel('FCN分数')
    plt.ylabel('频数')
    
    # 保存分布图
    plt.tight_layout()
    dist_path = os.path.join(args.output_dir, 'evaluation_metrics.png')
    plt.savefig(dist_path)
    
    # 添加到TensorBoard
    tb_writer.add_figure('Metrics/Distributions', plt.gcf(), 0)
    plt.close()
    
    print(f"评估指标分布图已保存: {dist_path}")
    print(f"测试完成! 查看结果: tensorboard --logdir={args.log_dir}")
    
    # 关闭TensorBoard
    tb_writer.close()


if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="CNN边缘图生成模型测试脚本")
    
    # 数据参数
    parser.add_argument("--test_dir", type=str, default="./processed/test",
                        help="测试数据目录")
    parser.add_argument("--output_dir", type=str, default="test_results",
                        help="结果保存目录")
    parser.add_argument("--image_size", type=int, default=256,
                        help="图像大小")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="批次大小")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="数据加载线程数")
    
    # 模型参数
    parser.add_argument("--model_path", type=str, 
                        default="./checkpoints/final_model.pth",
                        help="模型路径")
    parser.add_argument("--in_channels", type=int, default=1,
                        help="输入通道数")
    parser.add_argument("--out_channels", type=int, default=3,
                        help="输出通道数")
    parser.add_argument("--ngf", type=int, default=64,
                        help="特征图数量")
    
    # 测试参数
    parser.add_argument("--sample_freq", type=int, default=10,
                        help="保存样本频率")
    
    # TensorBoard参数
    parser.add_argument("--log_dir", type=str, default="runs",
                        help="TensorBoard日志目录")
    
    # 解析参数
    args = parser.parse_args()
    
    # 创建目录
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # 运行测试
    test_model(args) 