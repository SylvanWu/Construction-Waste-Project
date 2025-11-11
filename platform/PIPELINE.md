# Construction Waste Monitor - 核心Pipeline

## 🎯 主要处理管线

### **Pipeline 1: 数据集批处理**

```
输入目录 (604张图片)
    │
    ├─→ [初始化阶段]
    │   ├─→ 加载YOLO模型 (last.pt)
    │   ├─→ 加载体积模型 (best_strong_training.pth)
    │   ├─→ 加载基准图像 (frame_000009.png)
    │   └─→ 初始化核心模块 (Tracker, ROI, Counter, Volume, Visualizer)
    │
    ├─→ [处理循环] 对每一帧:
    │   │
    │   ├─→ 1. YOLO检测
    │   │   └─→ 输出: {boxes, masks, classes, confidences}
    │   │
    │   ├─→ 2. ROI更新
    │   │   └─→ 提取桶的区域mask
    │   │
    │   ├─→ 3. 物体跟踪
    │   │   ├─→ 距离匹配 (max_distance=50)
    │   │   ├─→ IoU匹配 (iou_threshold=0.3)
    │   │   ├─→ 重复检测 (防止重复计数)
    │   │   └─→ 输出: 跟踪物体列表
    │   │
    │   ├─→ 4. 入桶检测
    │   │   ├─→ ROI重叠检测 (overlap_threshold=0.1)
    │   │   ├─→ 中心点验证
    │   │   ├─→ 停留时间验证 (min_roi_frames=3)
    │   │   ├─→ 冷却期检查 (cooldown_frames=30)
    │   │   └─→ 输出: 新增计数
    │   │
    │   ├─→ 5. 体积估测
    │   │   ├─→ 深度学习推理 (448×448输入)
    │   │   ├─→ 相对体积计算
    │   │   └─→ 输出: 当前体积、填充率
    │   │
    │   ├─→ 6. 可视化
    │   │   ├─→ 绘制检测框
    │   │   ├─→ 绘制ROI区域
    │   │   ├─→ 绘制轨迹线
    │   │   ├─→ 绘制统计信息
    │   │   └─→ 保存可视化帧
    │   │
    │   └─→ 7. UI更新
    │       ├─→ 更新图像显示
    │       ├─→ 更新统计信息
    │       └─→ 更新进度条
    │
    └─→ [输出阶段]
        ├─→ 生成分析视频 (analysis_video.mp4)
        ├─→ 生成统计图表 (statistics_chart.png)
        ├─→ 保存JSON结果 (complete_results.json)
        └─→ 保存汇总报告 (summary_report.txt)
```

### **Pipeline 2: 实时摄像头处理**

```
摄像头输入 (实时流)
    │
    ├─→ [初始化阶段]
    │   ├─→ 打开摄像头 (camera_index=0)
    │   ├─→ 设置分辨率 (1280×720)
    │   ├─→ 设置FPS (30)
    │   └─→ 初始化核心模块
    │
    └─→ [实时处理循环]
        │
        ├─→ 读取一帧
        │
        ├─→ [相同的处理步骤1-6]
        │
        ├─→ 实时显示结果
        │
        └─→ 等待停止信号
```

## 🔑 关键参数配置

### **跟踪参数 (与Mix完全一致)**
```yaml
max_distance: 50          # 匹配最大距离(像素)
iou_threshold: 0.3        # IoU匹配阈值
disappear_frames: 5       # 消失判定帧数
use_iou_matching: true    # 启用IoU匹配
```

### **重复检测参数**
```python
# 距离阈值: max_distance * 2.0 = 100像素
# IoU阈值: iou_threshold * 0.3 = 0.09
# 时间窗口: 10帧
```

### **计数参数**
```yaml
overlap_threshold: 0.1    # ROI重叠比例阈值(10%)
use_center_point: true    # 启用中心点验证
min_roi_frames: 3         # 最小停留帧数(砖块/木头/纸板)
min_roi_frames: 1         # 最小停留帧数(瓶子/管道/塑料袋)
cooldown_frames: 30       # 冷却期(防止重复计数)
```

### **体积估测参数**
```yaml
input_size: 448           # 模型输入尺寸
max_volume: 100.0         # 最大容量(升)
baseline_volume: 1.04     # 空桶基准体积(升)
```

## 📊 输出结果结构

```
output_directory/
├── visualized_frames/
│   ├── frame_000001.png
│   ├── frame_000002.png
│   └── ... (604个可视化帧)
│
├── analysis_video.mp4        # 完整分析视频(10fps)
├── statistics_chart.png      # 统计图表
├── summary_report.txt        # 汇总报告
│
└── complete_results.json     # 完整结果
    ├── processing_summary
    │   ├── total_frames: 604
    │   └── processed_frames: 604
    │
    └── statistics
        ├── counting
        │   ├── total_objects: 40
        │   └── class_distribution
        │       ├── brick: 11
        │       ├── cardboard: 10
        │       ├── bottle: 7
        │       ├── wood: 6
        │       ├── plastic bag: 4
        │       └── pipe: 2
        │
        └── volume
            ├── current_volume: 134.75L
            ├── baseline_volume: 1.04L
            └── fill_percentage: 133.7%
```

## 🎯 与Mix系统的一致性保证

### **1. 核心算法一致**
- 直接导入MIx的核心模块
- 相同的处理流程
- 相同的参数配置

### **2. 数据结构一致**
- DetectedObject格式相同
- 跟踪结果格式相同
- 输出JSON格式相同

### **3. 性能一致**
- 相同的去重逻辑
- 相同的计数判定
- 相同的可视化生成

## ⚡ 性能指标

- **处理速度**: ~0.2秒/帧 (604帧约120秒)
- **准确率**: 与Mix系统一致 (40±2个物体)
- **内存占用**: ~2GB (模型加载+处理)
- **CPU使用**: 中等 (无GPU情况下)
- **GPU加速**: 支持CUDA (可选)

## 🔧 故障排除

### **问题1: 计数不准确**
- 检查max_distance参数 (应为50)
- 检查iou_threshold参数 (应为0.3)
- 检查disappear_frames参数 (应为5)

### **问题2: 性能慢**
- 启用GPU加速 (device: cuda)
- 降低可视化分辨率
- 减少体积估测频率

### **问题3: 内存不足**
- 减小batch_size
- 关闭实时可视化
- 使用较小的模型输入尺寸(224)

---

**Pipeline版本**: v1.0.0  
**测试数据集**: 604sequence  
**测试结果**: ✅ 通过 (40个物体，与Mix一致)

