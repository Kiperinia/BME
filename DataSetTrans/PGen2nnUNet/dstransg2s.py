import os
import json
import base64
import zlib
import cv2
import numpy as np
from tqdm import tqdm

def decode_supervisely_bitmap(b64_string):
    """解码 Supervisely Bitmap 字符串为二值 Mask [0, 1]"""
    try:
        zlib_bytes = base64.b64decode(b64_string)
        img_bytes = zlib.decompress(zlib_bytes)
        img_array = np.frombuffer(img_bytes, np.uint8)
        mask = cv2.imdecode(img_array, cv2.IMREAD_UNCHANGED)
        if mask is None: return None
        if len(mask.shape) == 3:
            mask_binary = (mask[:, :, 0] > 0).astype(np.uint8)
        else:
            mask_binary = (mask > 0).astype(np.uint8)
        return mask_binary
    except Exception as e:
        print(f"解码失败: {e}")
        return None

def get_bbox_from_mask(mask):
    """从二值 Mask 中提取 [xmin, ymin, xmax, ymax] 格式的 BBox"""
    y_indices, x_indices = np.where(mask > 0)
    if len(x_indices) == 0:
        return None  # 无病灶
    xmin = int(np.min(x_indices))
    ymin = int(np.min(y_indices))
    xmax = int(np.max(x_indices))
    ymax = int(np.max(y_indices))
    return [xmin, ymin, xmax, ymax]

def convert_polypgen_to_sam3(images_dir, json_dir, output_dir, target_size=(1024, 1024)):
    """将 PolypGen 转换为 SAM3 格式并生成提示文件"""
    # 1. 路径准备
    sam_images = os.path.join(output_dir, "images")
    sam_masks = os.path.join(output_dir, "masks")
    os.makedirs(sam_images, exist_ok=True)
    os.makedirs(sam_masks, exist_ok=True)

    # 存储所有图片的提示信息
    prompt_dict = {}

    json_files = [f for f in os.listdir(json_dir) if f.endswith('.json') and f != 'meta.json']
    print(f"找到 {len(json_files)} 个有效标注文件，开始转换...")

    for json_name in tqdm(json_files, desc="PolypGen to SAM3"):
        img_name = json_name.replace('.json', '')
        case_id = os.path.splitext(img_name)[0]
        
        img_path = os.path.join(images_dir, img_name)
        json_path = os.path.join(json_dir, json_name)
        
        if not os.path.exists(img_path): continue

        # --- 1. 处理原图 (Resizing) ---
        img = cv2.imread(img_path)
        h_orig, w_orig = img.shape[:2]
        # SAM3 通常需要固定的 $1024\times1024$ 输入
        img_resized = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
        # 保存缩放后的原图 (PNG 或高质量 JPG)
        save_img_name = f"{case_id}.png"
        cv2.imwrite(os.path.join(sam_images, save_img_name), img_resized)

        # --- 2. 处理 JSON 并生成全尺寸 Mask ---
        with open(json_path, 'r', encoding='utf-8') as f:
            ann_data = json.load(f)
            
        full_mask_orig = np.zeros((h_orig, w_orig), dtype=np.uint8)
        has_polyp = False
        
        for obj in ann_data.get('objects', []):
            if obj['classTitle'] == 'polyp' and 'bitmap' in obj:
                b64_data = obj['bitmap']['data']
                origin = obj['bitmap']['origin'] # [列(x), 行(y)]
                
                local_mask = decode_supervisely_bitmap(b64_data)
                if local_mask is not None:
                    has_polyp = True
                    lm_h, lm_w = local_mask.shape
                    x, y = origin[0], origin[1]
                    # 贴图
                    full_mask_orig[y:y+lm_h, x:x+lm_w] = np.maximum(full_mask_orig[y:y+lm_h, x:x+lm_w], local_mask)

        # --- 3. 缩放 Mask 并计算 BBox ---
        # 强制使用最近邻插值保持二值性
        mask_resized = cv2.resize(full_mask_orig, target_size, interpolation=cv2.INTER_NEAREST)
        # 保存 [0 1] 格式的掩码
        cv2.imwrite(os.path.join(sam_masks, f"{case_id}.png"), mask_resized)

        # 计算在 1024x1024 尺度下的 BBox (训练提示)
        bbox = get_bbox_from_mask(mask_resized)
        
        if bbox:
            # 记录提示信息：图片ID -> BBox
            prompt_dict[save_img_name] = bbox
        else:
            # 针对阴性样本，不生成 BBox 提示，或生成空提示，取决于仓库具体实现
            # 这里选择不写入，在 DataLoader 中手动处理
            pass

    # --- 4. 生成提示信息文件 prompts.json ---
    prompt_json_path = os.path.join(output_dir, "prompts.json")
    with open(prompt_json_path, "w", encoding='utf-8') as f:
        # indent=4 方便人阅读
        json.dump(prompt_dict, f, indent=4)
        
    print(f"SAM3 数据集准备完成！文件夹: {output_dir}")
    print(f"提示信息文件已生成: {prompt_json_path}")

if __name__ == "__main__":
    # 配置路径
    # 请修改为你实际存放 PolypGen 的原始目录
    IMAGES_DIR = r".\predata\PolypGen\img"
    JSON_DIR = r".\predata\PolypGen\ann"
    # 输出到全新的 SAM3 专属目录
    SAM3_OUTPUT_DIR = r".\afterdata\PolypGen-SAM3" 
    
    # 执行转换
    convert_polypgen_to_sam3(IMAGES_DIR, JSON_DIR, SAM3_OUTPUT_DIR, target_size=(1024, 1024))