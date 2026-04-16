import os
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt

def verify_single_case(case_id, dataset_dir, target_size=(1024, 1024)):
    """
    验证并可视化单个 SAM3 样本点
    case_id: 样本名称，不带后缀 (例如 'C1_100H0050')
    """
    img_path = os.path.join(dataset_dir, "images", f"{case_id}.png")
    mask_path = os.path.join(dataset_dir, "masks", f"{case_id}.png")
    prompt_file = os.path.join(dataset_dir, "prompts.json")

    # 1. 加载提示信息
    with open(prompt_file, 'r', encoding='utf-8') as f:
        prompts = json.load(f)
    
    img_key = f"{case_id}.png"
    if img_key not in prompts:
        print(f"❌ 在 prompts.json 中找不到 {img_key} 的记录")
        return

    bbox = prompts[img_key]
    print(f"检测到 {case_id} 的 BBox 坐标为: {bbox}")

    # 2. 读取并检查数据
    img = cv2.imread(img_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)

    if img is None or mask is None:
        print("❌ 读取图像或掩码失败，请检查文件是否存在")
        return

    # 维度检查
    print(f"图像维度: {img.shape[:2]} | 掩码维度: {mask.shape}")
    print(f"掩码唯一值 (应为 [0 1]): {np.unique(mask)}")

    # 3. 可视化
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    xmin, ymin, xmax, ymax = bbox

    plt.figure(figsize=(12, 6))

    # 左图：原图 + BBox
    plt.subplot(1, 2, 1)
    img_vis = img_rgb.copy()
    cv2.rectangle(img_vis, (xmin, ymin), (xmax, ymax), (255, 0, 0), 4) # 红色粗框
    plt.imshow(img_vis)
    plt.title(f"Image + BBox\n{img_key}")
    plt.axis('off')

    # 右图：Mask + BBox
    plt.subplot(1, 2, 2)
    # 将 1 放大到 255 以便肉眼观察
    mask_vis = cv2.cvtColor(mask * 255, cv2.COLOR_GRAY2RGB)
    cv2.rectangle(mask_vis, (xmin, ymin), (xmax, ymax), (0, 255, 0), 4) # 绿色粗框
    plt.imshow(mask_vis)
    plt.title("Mask + BBox Check")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 配置你的数据集路径
    DATASET_DIR = r".\afterdata\PolypGen-SAM3"
    
    # 填入你想查看的 case_id (不需要后缀)
    TEST_CASE = "C1_100H0050" 
    
    verify_single_case(TEST_CASE, DATASET_DIR)