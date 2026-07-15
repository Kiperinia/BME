# yolo_adapter/

YOLO 边界框适配器模块，将 YOLO 检测结果转换为 SAM3 可消费的边界框提示。支持检测缓存、CLI 参数集成和数据集构建。

## 文件说明

### `__init__.py`
模块入口，导出 `BBoxDetection`、`UltralyticsYoloDetector`、`YoloBoxProvider`、`create_box_provider`。

### `bbox_provider.py`
边界框提供器，将 YOLO 检测结果映射为 SAM3 输入坐标。

| 类/函数 | 说明 |
|---|---|
| `_record_key(record)` | 从记录字典提取唯一键值 |
| `_image_size(path)` | 获取图像尺寸 |
| `_scale_xyxy_to_square(xyxy, *, original_size, image_size)` | 将 xyxy 缩放到方形坐标系 |
| `YoloBoxProvider` | 基于 YOLO 检测的边界框提供器，支持缓存和回退策略 |
| `YoloBoxProvider.__init__(...)` | 初始化实例，加载缓存 |
| `YoloBoxProvider.get_box(record, image_size, ...)` | 获取记录对应的边界框张量 |
| `YoloBoxProvider._load_or_predict(key, image_path)` | 从缓存加载或调用 YOLO 预测 |
| `YoloBoxProvider._flush_cache()` | 将缓存写入磁盘 |
| `NoBoxProvider` | 始终返回哨兵值的提供器 |
| `NoBoxProvider.get_box(...)` | 返回已移除框的哨兵值 |
| `create_box_provider(...)` | 工厂函数，根据 source 创建相应提供器 |

### `cli.py`
CLI 辅助工具，用于在 SAM3 脚本中启用 YOLO 边界框提示。

| 函数 | 说明 |
|---|---|
| `add_yolo_bbox_args(parser)` | 向 ArgumentParser 添加 YOLO 框相关参数 |
| `build_box_provider_from_args(args, *, default_cache_name)` | 根据命令行参数构建边界框提供器 |

### `detector.py`
Ultralytics YOLO 检测器封装。

| 类/函数 | 说明 |
|---|---|
| `BBoxDetection` | 表示单个 YOLO 检测结果的数据类 |
| `BBoxDetection.to_dict()` | 转换为字典格式 |
| `UltralyticsYoloDetector` | YOLO 检测器封装 |
| `UltralyticsYoloDetector.__init__(...)` | 初始化检测器 |
| `UltralyticsYoloDetector._load_model()` | 加载 YOLO 模型 |
| `UltralyticsYoloDetector.predict_one(source)` | 对单张图像路径进行预测 |
| `UltralyticsYoloDetector.predict_one_array(image)` | 对 numpy 数组格式图像进行预测 |
| `UltralyticsYoloDetector._predict_source(source)` | 对任意源格式执行预测 |
| `UltralyticsYoloDetector._load_image_array(source)` | 加载图像为 numpy 数组 |

### `generate_bbox_cache.py`
为 MedEx-SAM3 分割文件生成 YOLO 边界框缓存。

| 函数 | 说明 |
|---|---|
| `main()` | 命令行入口，读取分割记录，运行 YOLO 检测并输出 JSON 缓存 |

### `prepare_yolo_dataset_from_splits.py`
从 MedEx-SAM3 分割记录构建 YOLO 检测数据集。将分割掩码转换为填充边界框标签。

| 函数 | 说明 |
|---|---|
| `_safe_stem(value, fallback)` | 将字符串转换为安全的文件系统名称 |
| `_record_key(record, index)` | 生成唯一且安全的文件键名 |
| `_load_rgb_image(path)` | 加载 RGB 图像，支持 _0000 多通道格式 |
| `_mask_to_xyxy(mask_path)` | 将掩码图像转换为 xyxy 坐标 |
| `_pad_xyxy(xyxy, *, width, height, ...)` | 对边界框应用填充并确保最小尺寸 |
| `_to_yolo_line(xyxy, width, height)` | 将 xyxy 转换为 YOLO 格式标签行 |
| `_materialize_image(source, destination, *, link_mode)` | 复制/链接源图像到目标路径 |
| `_prepare_split(...)` | 处理单个数据集划分 |
| `_write_data_yaml(output_dir, splits)` | 写入 YOLO data.yaml 配置文件 |
| `main()` | 命令行入口，转换分割记录为 YOLO 数据集 |
