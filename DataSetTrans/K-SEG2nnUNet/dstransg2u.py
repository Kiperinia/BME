import os
import json
import base64
import zlib
import cv2
import numpy as np
from tqdm import tqdm

def decode_supervisely_bitmap(b64_string):
    """
    将 Supervisely 的 Base64 位图字符串解码为 numpy 数组 (Mask)
    """
    # 1. Base64 解码
    zlib_bytes = base64.b64decode(b64_string)
    # 2. zlib 解压缩
    img_bytes = zlib.decompress(zlib_bytes)
    # 3. 转换为 numpy 数组并使用 cv2 解码
    img_array = np.frombuffer(img_bytes, np.uint8)
    mask = cv2.imdecode(img_array, cv2.IMREAD_UNCHANGED)
    
    # 提取二值化掩码
    if len(mask.shape) == 3:
        # 如果带有透明度通道或RGB，通常取第一个通道即可判断
        mask_binary = (mask[:, :, 0] > 0).astype(np.uint8)
    else:
        mask_binary = (mask > 0).astype(np.uint8)
        
    return mask_binary

def convert_polypgen_to_nnunet(images_dir, json_dir, output_dir, dataset_id=502, dataset_name="PolypGen"):
    task_name = f"Dataset{dataset_id:03d}_{dataset_name}"
    out_imagesTr = os.path.join(output_dir, task_name, "imagesTr")
    out_labelsTr = os.path.join(output_dir, task_name, "labelsTr")
    
    os.makedirs(out_imagesTr, exist_ok=True)
    os.makedirs(out_labelsTr, exist_ok=True)

    json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]
    print(f"找到 {len(json_files)} 个标注文件，开始转换...")

    for json_name in tqdm(json_files, desc="Converting PolypGen"):
        # PolypGen 的 json 文件名通常是: 图像名.json，例如 C1_100H0050.jpg.json
        img_name = json_name.replace('.json', '')
        case_id = os.path.splitext(img_name)[0] # 去掉扩展名，如 C1_100H0050
        
        img_path = os.path.join(images_dir, img_name)
        json_path = os.path.join(json_dir, json_name)
        
        if not os.path.exists(img_path):
            print(f"警告: 找不到对应的原图 {img_path}，跳过该文件")
            continue

        # --- 1. 处理 JSON 生成 Mask ---
        with open(json_path, 'r', encoding='utf-8') as f:
            ann_data = json.load(f)
            
        height = ann_data['size']['height']
        width = ann_data['size']['width']
        
        # 初始化一张全尺寸的全黑背景图
        full_mask = np.zeros((height, width), dtype=np.uint8)
        
        has_polyp = False
        for obj in ann_data.get('objects', []):
            if obj['classTitle'] == 'polyp' and 'bitmap' in obj:
                has_polyp = True
                b64_data = obj['bitmap']['data']
                origin = obj['bitmap']['origin'] # [x, y] 即 [列, 行]
                
                # 解码局部息肉的 Mask
                local_mask = decode_supervisely_bitmap(b64_data)
                
                # 计算贴图的边界
                h, w = local_mask.shape
                x, y = origin[0], origin[1]
                
                # 将局部息肉 Mask 贴到全图对应位置上
                # 由于 nnU-Net 类别是 1，我们直接用 local_mask (全是 0 和 1) 进行覆盖
                # 使用 np.maximum 处理可能存在的多个息肉重叠的情况
                full_mask[y:y+h, x:x+w] = np.maximum(full_mask[y:y+h, x:x+w], local_mask)

        if not has_polyp:
            # 如果这是一个阴性样本（没有息肉），就直接保存全黑背景
            pass

        # 保存为 nnU-Net 格式的 png 标签
        cv2.imwrite(os.path.join(out_labelsTr, f"{case_id}.png"), full_mask)

        # --- 2. 处理原图 (RGB 拆分) ---
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        for i in range(3):
            cv2.imwrite(os.path.join(out_imagesTr, f"{case_id}_{i:04d}.png"), img[:, :, i])

    # --- 3. 生成 dataset.json ---
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
        "numTraining": len(json_files),
        "file_ending": ".png"
    }
    
    with open(os.path.join(output_dir, task_name, "dataset.json"), "w") as f:
        json.dump(dataset_json, f, indent=4)
        
    print(f"PolypGen 转换完成！请使用命令进行预处理: nnUNetv2_plan_and_preprocess -d {dataset_id}")

if __name__ == "__main__":
    # 请修改为你存放 PolypGen 的实际目录
    # 假设你的目录结构是：
    # PolypGen_data/
    # ├── images/ (存放 C1_100H0050.jpg 等)
    # └── ann/    (存放 C1_100H0050.jpg.json 等)
    
    IMAGES_DIR = r".\predata\PolypGen\img"
    JSON_DIR = r".\predata\PolypGen\ann"
    NNUNET_RAW_DIR = r".\afterdata\PolypGen-nnunet_raw" 
    
    convert_polypgen_to_nnunet(IMAGES_DIR, JSON_DIR, NNUNET_RAW_DIR)