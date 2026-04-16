import os
import cv2
import numpy as np
# 请替换为你转换后的实际路径
test_path = 'afterdata\PolypGen-nnunet_raw\Dataset502_PolypGen\labelsTr\C1_100H0050.png'

if os.path.exists(test_path):
    print(f"✅ 找到文件了: {test_path}")
    mask = cv2.imread(test_path, cv2.IMREAD_UNCHANGED)
    if mask is not None:
        print(f"📊 唯一值为: {np.unique(mask)}")
    else:
        print("❌ 文件存在但无法解码（可能是格式损坏）")
else:
    print(f"🚫 文件不存在，请检查路径: {os.path.abspath(test_path)}")