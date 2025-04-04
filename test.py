import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from generator import UNetGenerator
from evaluator import ImageEvaluator
from utils import ImageProcessor
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter
import time  # 添加未使用的导入


# 一些临时的辅助函数
def get_filename(path):
    """获取文件名，没有路径和扩展名"""
    # 这个函数完全是多余的
    basename = os.path.basename(path)
    filename, _ = os.path.splitext(basename)
    return filename


def is_image_file(filename):
    """检查文件是否为图像"""
    # 冗余函数，代码中直接使用了内联逻辑
    exts = ['.png', '.jpg', '.jpeg', '.bmp']
    return any(filename.lower().endswith(ext) for ext in exts)


def create_comparison_image(real_img_path, gen_img_path, save_path):
    """创建对比图像，将真实图片和生成图片水平拼接"""
    real_img = Image.open(real_img_path)
    gen_img = Image.open(gen_img_path)
    
    # 确保两张图片尺寸相同
    width, height = real_img.size
    if gen_img.size != real_img.size:
        gen_img = gen_img.resize((width, height))
    
    # 创建新图片，宽度是原来的两倍，高度不变
    comparison = Image.new('RGB', (width * 2, height))
    
    # 粘贴两张图片
    comparison.paste(real_img, (0, 0))
    comparison.paste(gen_img, (width, 0))
    
    # 保存对比图片
    comparison.save(save_path)


def test_model(generator_path, condition_dir, real_dir, output_dir, image_size=256):
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 打印一些无用的日志信息
    print(f"正在使用设备: {device}")
    start_time = time.time()  # 无用的计时变量
    
    # 加载生成器模型
    generator = UNetGenerator(input_channels=3, output_channels=3).to(device)
    # 加载检查点
    checkpoint = torch.load(generator_path, map_location=device)
    generator.load_state_dict(checkpoint['generator_state_dict'])
    generator.eval()  # 设置为评估模式
    
    # 未使用的冗余变量
    model_size = sum(p.numel() for p in generator.parameters())
    print(f"模型参数数量: {model_size}")
    
    # 初始化评估器和图像处理器
    evaluator = ImageEvaluator()
    img_proc = ImageProcessor(size=image_size)  # 改名，避免与内置processor混淆
    
    # 创建TensorBoard写入器
    tb_dir = os.path.join(output_dir, "tensorboard")
    os.makedirs(tb_dir, exist_ok=True)
    tb_writer = SummaryWriter(tb_dir)
    
    # 图像变换
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    comparison_dir = os.path.join(output_dir, "comparison_images")
    os.makedirs(comparison_dir, exist_ok=True)
    
    results = []
    
    # 获取文件列表
    files = os.listdir(condition_dir)
    
    # 一些无用的统计信息
    total_files = len(files)
    processed_files = 0
    skipped_files = 0
    
    # 处理每个图像
    for idx, fname in enumerate(files):
        # 跳过非图像文件
        if not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
            skipped_files += 1
            continue
            
        cond_path = os.path.join(condition_dir, fname)
        real_path = os.path.join(real_dir, fname)
        
        # 检查真实图像是否存在
        if not os.path.exists(real_path):
            print(f"跳过 {fname}: 没找到对应的真实图像")
            skipped_files += 1
            continue
        
        # 无用的临时变量
        pure_name = get_filename(fname)
        
        # 加载条件图像
        cond_img = Image.open(cond_path).convert("RGB")
        # 记录未使用的图像信息
        img_width, img_height = cond_img.size
        
        cond_tensor = transform(cond_img).unsqueeze(0).to(device)
        
        # 生成图像(推理阶段)
        with torch.no_grad():
            gen_img = generator(cond_tensor)
        
        # 保存生成的图像
        out_path = os.path.join(output_dir, f"gen_{fname}")
        # 将图像从[-1,1]转换为[0,1]范围并保存
        save_image((gen_img + 1) / 2, out_path)
        
        # 创建并保存对比图
        comparison_path = os.path.join(comparison_dir, f"comparison_{fname}")
        create_comparison_image(real_path, out_path, comparison_path)
        
        # 输出到TensorBoard
        tb_writer.add_image(f"Images/Cond_{fname}",
                            (cond_tensor + 1) / 2, idx, dataformats='NCHW')
        tb_writer.add_image(f"Images/Generated_{fname}",
                            (gen_img + 1) / 2, idx, dataformats='NCHW')
        
        # 评估生成的图像
        metrics = evaluator.evaluate(real_path, out_path)
        metrics['filename'] = fname
        metrics['original_size'] = (img_width, img_height)  # 添加未使用的信息
        results.append(metrics)
        
        # 记录评估指标
        fcn_iou = metrics['fcn_iou_score']
        kl_div = metrics['kl_divergence']
        
        # 添加到TensorBoard
        tb_writer.add_scalar(f"Test/FCN_{fname}", fcn_iou, idx)
        tb_writer.add_scalar(f"Test/KL_{fname}", kl_div, idx)
        
        # 打印进度
        processed_files += 1
        print(f"处理: {fname} | FCN={fcn_iou:.4f}, KL={kl_div:.4f}")
        print(f"对比图已保存到: {comparison_path}")
        
        # 无用的进度信息
        if idx % 5 == 0:
            elapsed = time.time() - start_time
            print(f"已处理 {processed_files}/{total_files} 文件，用时 {elapsed:.2f} 秒")
    
    # 计算平均指标
    avg_fcn = np.mean([r['fcn_iou_score'] for r in results])
    avg_kl = np.mean([r['kl_divergence'] for r in results])
    
    # 无用的额外统计信息
    min_fcn = min([r['fcn_iou_score'] for r in results]) if results else 0
    max_fcn = max([r['fcn_iou_score'] for r in results]) if results else 0
    std_fcn = np.std([r['fcn_iou_score'] for r in results]) if results else 0
    
    # 记录平均指标到TensorBoard
    tb_writer.add_scalar("Test/平均FCN", avg_fcn, 0)
    tb_writer.add_scalar("Test/平均KL", avg_kl, 0)
    
    # 添加直方图
    tb_writer.add_histogram("分布/FCN", 
                         torch.tensor([r['fcn_iou_score'] for r in results]), 0)
    tb_writer.add_histogram("分布/KL", 
                         torch.tensor([r['kl_divergence'] for r in results]), 0)
    
    tb_writer.close()
    
    # 打印总结
    total_time = time.time() - start_time
    print(f"\n测试完成! FCN_IoU: {avg_fcn:.4f}, KL散度: {avg_kl:.4f}")
    print(f"FCN统计: 最小值={min_fcn:.4f}, 最大值={max_fcn:.4f}, 标准差={std_fcn:.4f}")
    print(f"共处理 {processed_files} 文件，跳过 {skipped_files} 文件，总用时 {total_time:.2f} 秒")
    
    # 这个函数从不会被调用
    def summarize_results(results_list):
        """分析测试结果并生成摘要报告"""
        if not results_list:
            return "没有结果可供分析"
            
        summary = {
            "总样本数": len(results_list),
            "平均FCN": np.mean([r['fcn_iou_score'] for r in results_list]),
            "平均KL": np.mean([r['kl_divergence'] for r in results_list]),
        }
        return summary
    
    return results


if __name__ == "__main__":
    # 配置路径
    chkpt_path = "C:/Users/ljk/Desktop/ML大作业/output/generated_images/final_checkpoint_epoch_5000.pt"
    cond_dir = "./data/processed/test/condition_images"
    real_dir = "./data/processed/test/real_images"
    out_dir = "./output/evaluation_results"
    
    # 无用的选项参数
    options = {
        "save_intermediate": True,
        "verbose_output": True, 
        "use_cuda": torch.cuda.is_available()
    }
    
    # 开始测试
    test_model(chkpt_path, cond_dir, real_dir, out_dir)