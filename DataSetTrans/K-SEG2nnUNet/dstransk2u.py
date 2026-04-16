import os
import cv2
import json
import numpy as np
from pathlib import Path


def convert_kvasir_to_nnunet(kvasir_images_dir, kvasir_masks_dir, nnunet_dataset_dir, dataset_id=501,
                             dataset_name="KvasirSEG"):
    """
    将 Kvasir-SEG 数据集转换为 nnU-Net v2 格式
    """
    # 定义 nnU-Net 的输出目录
    task_name = f"Dataset{dataset_id:03d}_{dataset_name}"
    out_base = os.path.join(nnunet_dataset_dir, task_name)
    out_imagesTr = os.path.join(out_base, "imagesTr")
    out_labelsTr = os.path.join(out_base, "labelsTr")

    # 创建目录
    os.makedirs(out_imagesTr, exist_ok=True)
    os.makedirs(out_labelsTr, exist_ok=True)

    # 获取所有图像文件名 (Kvasir 的图片通常是 .jpg，mask 可能也是 .jpg 或 .png)
    image_files = [f for f in os.listdir(kvasir_images_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]

    print(f"找到 {len(image_files)} 张图像，开始转换...")

    for img_name in image_files:
        case_id = os.path.splitext(img_name)[0]

        # 1. 处理原始图像 (拆分 RGB 通道)
        img_path = os.path.join(kvasir_images_dir, img_name)
        img = cv2.imread(img_path)
        if img is None:
            print(f"警告: 无法读取图像 {img_path}")
            continue

        # OpenCV 默认读取为 BGR，转换为 RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 分离 R, G, B 通道，并保存为 _0000, _0001, _0002
        cv2.imwrite(os.path.join(out_imagesTr, f"{case_id}_0000.png"), img[:, :, 0])  # R 通道
        cv2.imwrite(os.path.join(out_imagesTr, f"{case_id}_0001.png"), img[:, :, 1])  # G 通道
        cv2.imwrite(os.path.join(out_imagesTr, f"{case_id}_0002.png"), img[:, :, 2])  # B 通道

        # 2. 处理标签 Mask (二值化：255 -> 1)
        # Kvasir的mask文件名和原图相同，但后缀可能不同，这里兼容 .jpg 和 .png
        mask_path_jpg = os.path.join(kvasir_masks_dir, f"{case_id}.jpg")
        mask_path_png = os.path.join(kvasir_masks_dir, f"{case_id}.png")

        if os.path.exists(mask_path_jpg):
            mask_path = mask_path_jpg
        elif os.path.exists(mask_path_png):
            mask_path = mask_path_png
        else:
            print(f"警告: 找不到 {case_id} 对应的 mask 文件")
            continue

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # 将大于 0 的像素点归为 1 (息肉类别)
        _, mask_binary = cv2.threshold(mask, 127, 1, cv2.THRESH_BINARY)

        # 保存 mask (nnU-Net 要求 .png 格式)
        cv2.imwrite(os.path.join(out_labelsTr, f"{case_id}.png"), mask_binary)

    # 3. 生成 dataset.json
    print("生成 dataset.json ...")
    dataset_json = {
        "channel_names": {
            "0": "R",
            "1": "G",
            "2": "B"
        },
        "labels": {
            "background": 0,
            "polyp": 1
        },
        "numTraining": len(image_files),
        "file_ending": ".png"
    }

    with open(os.path.join(out_base, "dataset.json"), "w") as f:
        json.dump(dataset_json, f, indent=4)

    print(f"转换完成！数据集已保存至: {out_base}")


if __name__ == "__main__":
    # ================= 配置路径 =================
    # Kvasir-SEG 原图路径
    KVASIR_IMAGES_DIR = "./predata/Kvasir-SEG/images"
    # Kvasir-SEG 标签路径
    KVASIR_MASKS_DIR = "./predata/Kvasir-SEG/masks"
    # nnU-Net 原始数据存放的根目录 (即 nnUNet_raw 环境变量指向的目录)
    NNUNET_RAW_DIR = "./afterdata/Kvasir-SEG-nnunet_raw"

    # 执行转换
    convert_kvasir_to_nnunet(
        kvasir_images_dir=KVASIR_IMAGES_DIR,
        kvasir_masks_dir=KVASIR_MASKS_DIR,
        nnunet_dataset_dir=NNUNET_RAW_DIR,
        dataset_id=501,  # 你可以自定义 ID，建议 >= 500 以区分医学 3D 数据
        dataset_name="KvasirSEG"
    )