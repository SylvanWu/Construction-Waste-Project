# Construction Waste Monitor - 架构与调用关系

## 📊 完整调用关系图

```
┌─────────────────────────────────────────────────────────────────┐
│                        main.py (入口)                            │
│  - setup_logger()                                               │
│  - QApplication 初始化                                           │
│  - 样式表加载                                                     │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                   app/main_window.py                            │
│                     MainWindow                                  │
│                                                                 │
│  ┌──────────────────────────────────────────────────────┐     │
│  │  UI组件初始化                                          │     │
│  │  - create_tab_bar()    → 水平标签栏                   │     │
│  │  - create_left_panel() → 左侧设置面板                 │     │
│  │  - create_right_panel()→ 右侧结果面板                 │     │
│  └──────────────────────────────────────────────────────┘     │
│                                                                 │
│  ┌──────────────────────────────────────────────────────┐     │
│  │  核心功能方法                                          │     │
│  │  - start_processing()                                │     │
│  │  - prepare_config()                                  │     │
│  │  - on_frame_processed()                              │     │
│  │  - on_processing_finished()                          │     │
│  └──────────────────────────────────────────────────────┘     │
└───────┬─────────────────┬─────────────────┬───────────────────┘
        │                 │                 │
        ▼                 ▼                 ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│ ConfigManager│  │ProcessingThread│ │  UI Widgets  │
└──────┬───────┘  └───────┬────────┘  └──────┬───────┘
       │                  │                   │
       ▼                  ▼                   ▼
```

## 🔄 主要调用流程

### 1️⃣ **应用启动流程**

```
main.py::main()
  │
  ├─→ setup_logger()
  │   └─→ loguru.add(log_file, rotation="10MB", level="DEBUG")
  │
  ├─→ QApplication(sys.argv)
  │   ├─→ setApplicationName("Construction Waste Monitor")
  │   ├─→ setStyleSheet(qss_content)
  │   └─→ exec()
  │
  └─→ MainWindow()
      ├─→ __init__()
      │   ├─→ ConfigManager()
      │   └─→ init_ui()
      │       ├─→ create_menus()
      │       ├─→ create_tab_bar()
      │       ├─→ create_left_panel()
      │       │   ├─→ ModelSelector()
      │       │   ├─→ DatasetBrowser()
      │       │   ├─→ CameraSelector()
      │       │   └─→ ParamPanel()
      │       └─→ create_right_panel()
      │           └─→ ResultViewer()
      │
      └─→ load_default_models()
```

### 2️⃣ **数据集处理流程**

```
MainWindow.start_processing()
  │
  ├─→ prepare_config()
  │   ├─→ ConfigManager.to_dict()
  │   ├─→ ModelSelector.get_model_paths()
  │   ├─→ ParamPanel.get_params()
  │   └─→ CameraSelector.get_settings()
  │
  ├─→ DatasetProcessor(config_dict, input_dir, output_dir)
  │   │
  │   └─→ MixProcessor.__init__()
  │       │
  │       ├─→ _parse_config()
  │       │   └─→ 解析所有配置参数
  │       │
  │       ├─→ _init_models()
  │       │   ├─→ DetectionModel(yolo_path, device, confidence)
  │       │   │   └─→ YOLO11加载
  │       │   │
  │       │   └─→ VolumeModel(checkpoint_path, device)
  │       │       └─→ PyTorch模型加载
  │       │
  │       └─→ _init_core_modules()
  │           ├─→ ObjectTracker(max_distance, iou_threshold, ...)
  │           ├─→ ROIManager(bin_class_id, overlap_threshold, ...)
  │           ├─→ ObjectCounter(class_names, output_dir, ...)
  │           ├─→ VolumeEstimator(volume_model, baseline_image)
  │           └─→ Visualizer(config)
  │
  ├─→ ProcessingThread(processor, mode='dataset')
  │   │
  │   └─→ run()
  │       └─→ processor.process_dataset(progress_callback, stop_event)
  │
  └─→ [信号连接]
      ├─→ progress_updated → on_progress_updated()
      ├─→ frame_processed → on_frame_processed()
      └─→ finished → on_processing_finished()
```

### 3️⃣ **单帧处理流程**

```
DatasetProcessor.process_dataset()
  │
  ├─→ load_images()
  │   └─→ glob + 去重
  │
  ├─→ reset()
  │   ├─→ tracker.reset()
  │   ├─→ counter.reset_counts()
  │   └─→ volume_estimator.reset()
  │
  └─→ [循环处理每一帧]
      │
      └─→ process_frame(frame, frame_id, do_visualization=True)
          │
          ├─→ 1. 物体检测
          │   │
          │   └─→ detection_model.detect(frame)
          │       └─→ YOLO.predict()
          │           └─→ 返回 {boxes, masks, classes, confidences}
          │
          ├─→ 2. ROI更新
          │   │
          │   └─→ roi_manager.update_roi(detection_results, frame.shape)
          │       ├─→ _find_bin_mask()
          │       └─→ _calculate_roi_properties()
          │
          ├─→ 3. 物体跟踪
          │   │
          │   └─→ _update_tracking(detection_results)
          │       │
          │       └─→ tracker.update(tracker_input)
          │           │
          │           ├─→ _create_detected_objects()
          │           │
          │           ├─→ _match_and_update(detected_objects)
          │           │   ├─→ _calculate_cost_matrix()
          │           │   │   ├─→ _calculate_distance_cost()
          │           │   │   └─→ _calculate_iou_cost()
          │           │   │
          │           │   ├─→ _hungarian_matching()
          │           │   │
          │           │   └─→ [对未匹配物体]
          │           │       ├─→ _is_duplicate_object()
          │           │       │   ├─→ 类别检查
          │           │       │   ├─→ 距离检查 (max_distance*2.0)
          │           │       │   ├─→ IoU检查 (iou_threshold*0.3)
          │           │       │   ├─→ 时间窗口检查 (<10帧)
          │           │       │   └─→ 额外相似度检查
          │           │       │
          │           │       └─→ [如非重复] _get_next_id()
          │           │
          │           └─→ _cleanup_lost_objects()
          │               └─→ 移除消失超过disappear_frames的物体
          │
          ├─→ 4. 计数统计
          │   │
          │   └─→ counter.update_counts(tracked_objects, roi_manager, frame_count)
          │       │
          │       └─→ [遍历每个物体]
          │           │
          │           ├─→ roi_manager.is_object_in_roi(obj)
          │           │   ├─→ _check_overlap_area()
          │           │   │   └─→ mask重叠比例计算
          │           │   │
          │           │   └─→ _check_center_point()
          │           │       └─→ 中心点位置验证
          │           │
          │           ├─→ [计数条件判断]
          │           │   ├─→ 未被计数
          │           │   ├─→ 在ROI中
          │           │   ├─→ 在ROI时间 >= min_roi_frames
          │           │   └─→ 冷却期检查
          │           │
          │           └─→ [满足条件] _count_object()
          │               ├─→ counted_objects.add(obj.id)
          │               ├─→ total_count += 1
          │               ├─→ class_counts[class_name] += 1
          │               └─→ counting_history.append()
          │
          ├─→ 5. 体积估测
          │   │
          │   └─→ volume_estimator.estimate_volume(frame, roi_mask)
          │       ├─→ volume_model.estimate(frame)
          │       │   ├─→ 图像预处理
          │       │   ├─→ PyTorch推理
          │       │   └─→ 后处理
          │       │
          │       └─→ 计算相对体积和填充率
          │
          ├─→ 6. 可视化生成
          │   │
          │   └─→ visualizer.draw_frame(frame, ...)
          │       ├─→ 绘制检测框
          │       ├─→ 绘制ROI区域
          │       ├─→ 绘制轨迹
          │       ├─→ 绘制统计信息
          │       └─→ 绘制体积信息
          │
          └─→ [返回结果]
              ├─→ frame_id
              ├─→ frame_index
              ├─→ visualized_frame
              ├─→ current_counts
              ├─→ volume_info
              └─→ has_new_object
```

### 4️⃣ **UI更新流程**

```
ProcessingThread.run()
  │
  ├─→ [每帧处理完成]
  │   │
  │   └─→ emit: frame_processed(result)
  │       │
  │       └─→ MainWindow.on_frame_processed(result)
  │           │
  │           ├─→ ResultViewer.update_image(visualized_frame)
  │           │   ├─→ cv2.cvtColor(BGR→RGB)
  │           │   ├─→ QImage创建
  │           │   ├─→ QPixmap缩放
  │           │   └─→ QLabel.setPixmap()
  │           │
  │           └─→ ResultViewer.update_stats(result)
  │               └─→ QTextEdit.setPlainText()
  │
  ├─→ [进度更新]
  │   │
  │   └─→ emit: progress_updated(current, total, result)
  │       │
  │       └─→ MainWindow.on_progress_updated()
  │           ├─→ QProgressBar.setValue()
  │           └─→ statusBar().showMessage()
  │
  └─→ [处理完成]
      │
      └─→ emit: finished(results)
          │
          └─→ MainWindow.on_processing_finished(results)
              ├─→ QProgressBar隐藏
              ├─→ 显示统计信息
              ├─→ ResultViewer.processing_complete()
              └─→ QMessageBox.information()
```

## 🏗️ 模块依赖关系

### **核心依赖链**

```
main.py
  │
  ├─→ app/
  │   ├─→ main_window.py
  │   │   ├─→ PyQt6 (UI框架)
  │   │   ├─→ platform_core (业务逻辑)
  │   │   └─→ app.widgets (UI组件)
  │   │
  │   └─→ widgets/
  │       ├─→ model_selector.py    → PyQt6
  │       ├─→ dataset_browser.py   → PyQt6
  │       ├─→ camera_selector.py   → PyQt6 + OpenCV
  │       ├─→ param_panel.py       → PyQt6
  │       └─→ result_viewer.py     → PyQt6 + OpenCV + NumPy
  │
  └─→ platform_core/
      ├─→ config_manager.py        → YAML + Loguru
      │
      └─→ processor.py
          │
          ├─→ MIx/models/
          │   ├─→ detection_model.py    → YOLO11 + PyTorch
          │   └─→ volume_model.py       → PyTorch + Torchvision
          │
          ├─→ MIx/core/
          │   ├─→ object_tracker.py     → NumPy + SciPy
          │   ├─→ roi_manager.py        → OpenCV + NumPy
          │   ├─→ counter.py            → Python标准库
          │   └─→ volume_estimator.py   → NumPy
          │
          └─→ MIx/utils/
              └─→ visualizer.py         → OpenCV + Matplotlib
```

### **第三方库依赖**

```
┌─────────────────┐
│   UI层          │
├─────────────────┤
│ PyQt6           │  → 桌面应用框架
└─────────────────┘
        │
        ▼
┌─────────────────┐
│   业务逻辑层     │
├─────────────────┤
│ Loguru          │  → 日志系统
│ YAML            │  → 配置管理
│ JSON            │  → 数据序列化
└─────────────────┘
        │
        ▼
┌─────────────────┐
│   深度学习层     │
├─────────────────┤
│ PyTorch         │  → 深度学习框架
│ Torchvision     │  → 计算机视觉工具
│ Ultralytics     │  → YOLO11实现
└─────────────────┘
        │
        ▼
┌─────────────────┐
│   计算机视觉层   │
├─────────────────┤
│ OpenCV          │  → 图像处理
│ NumPy           │  → 数值计算
│ SciPy           │  → 科学计算
└─────────────────┘
        │
        ▼
┌─────────────────┐
│   可视化层      │
├─────────────────┤
│ Matplotlib      │  → 图表绘制
│ Seaborn         │  → 统计图表
└─────────────────┘
```

## 🎯 关键算法流程

### **重复物体检测算法**

```python
def _is_duplicate_object(new_obj, existing_objects):
    for existing_obj in existing_objects:
        # 1. 类别过滤
        if existing_obj.class_id != new_obj.class_id:
            continue
        
        # 2. 距离检查 (放宽到2倍max_distance)
        distance = existing_obj.distance_to(new_obj.center)
        if distance > max_distance * 2.0:  # 100像素
            continue
        
        # 3. IoU检查 (放宽到30%阈值)
        iou = existing_obj.calculate_iou(new_obj.bbox)
        if iou < iou_threshold * 0.3:  # 0.09
            continue
        
        # 4. 时间窗口检查 (扩大到10帧)
        frames_since_seen = current_frame - existing_obj.last_seen_frame
        if frames_since_seen < 10:
            return True  # 判定为重复
        
        # 5. 高相似度检查
        if distance < max_distance * 0.5 and iou > iou_threshold * 0.5:
            return True
    
    return False
```

### **入桶检测算法**

```python
def update_counts(objects, roi_manager, current_frame):
    for obj in objects:
        # 1. 跳过桶本身
        if obj.class_id == bin_class_id:
            continue
        
        # 2. ROI检测
        in_roi = roi_manager.is_object_in_roi(obj)
        #   ├─→ 重叠面积检测 (overlap_ratio >= 0.1)
        #   └─→ 中心点检测 (center in roi_mask)
        
        # 3. 记录进入时间
        if in_roi and obj.roi_entry_frame == -1:
            obj.roi_entry_frame = current_frame
        
        # 4. 计数条件判断
        should_count = (
            obj.id not in counted_objects and       # 未计数
            obj.roi_entry_frame != -1 and           # 已进入ROI
            (current_frame - obj.roi_entry_frame) >= min_roi_frames  # 停留足够时间
        )
        
        # 5. 冷却期检查 (防止重复计数)
        if should_count and obj.counted_frame != -1:
            if (current_frame - obj.counted_frame) < cooldown_frames:
                should_count = False
        
        # 6. 执行计数
        if should_count:
            _count_object(obj, current_frame)
```

## 📊 数据流向图

```
┌──────────────┐
│  输入源       │
│ (摄像头/图片) │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  图像数据     │
│  (BGR格式)    │
└──────┬───────┘
       │
       ▼
┌──────────────┐     ┌──────────────┐
│ YOLO检测     │────→│ 检测结果      │
│              │     │ {boxes, masks}│
└──────────────┘     └──────┬───────┘
                            │
                            ▼
                     ┌──────────────┐
                     │ ROI管理      │
                     │ (桶区域提取)  │
                     └──────┬───────┘
                            │
                            ▼
                     ┌──────────────┐     ┌──────────────┐
                     │ 物体跟踪      │────→│ 跟踪物体列表  │
                     │ (去重+匹配)   │     │ [obj1, obj2] │
                     └──────────────┘     └──────┬───────┘
                                                 │
                                                 ▼
                                          ┌──────────────┐
                                          │ 计数统计      │
                                          │ (入桶检测)    │
                                          └──────┬───────┘
                                                 │
       ┌─────────────────────────────────────────┴──────────┐
       │                                                    │
       ▼                                                    ▼
┌──────────────┐                                    ┌──────────────┐
│ 体积估测      │                                    │ 可视化生成    │
│ (深度学习)    │                                    │ (OpenCV绘图)  │
└──────┬───────┘                                    └──────┬───────┘
       │                                                    │
       └────────────────────┬───────────────────────────────┘
                            │
                            ▼
                     ┌──────────────┐
                     │ 结果输出      │
                     │ - JSON文件    │
                     │ - 可视化帧    │
                     │ - 统计图表    │
                     │ - 分析视频    │
                     └──────────────┘
```

## 🚀 启动到结果完整流程

```
[1] 用户启动应用
    python main.py
    
[2] 初始化阶段
    ├─→ 创建日志系统
    ├─→ 初始化PyQt6应用
    ├─→ 加载样式表
    └─→ 显示主窗口
    
[3] 用户配置
    ├─→ [模型设置标签页]
    │   ├─→ 选择YOLO模型
    │   ├─→ 选择体积模型
    │   └─→ 选择基准图像
    │
    ├─→ [数据集处理标签页]
    │   ├─→ 选择输入目录
    │   └─→ 选择输出目录
    │
    └─→ [参数标签页]
        ├─→ 设置跟踪参数
        ├─→ 设置计数参数
        └─→ 设置体积参数
        
[4] 开始处理
    ├─→ 点击"Start Processing"按钮
    ├─→ 创建配置字典
    ├─→ 初始化处理器
    │   ├─→ 加载YOLO模型
    │   ├─→ 加载体积模型
    │   └─→ 初始化核心模块
    │
    └─→ 创建处理线程
    
[5] 逐帧处理
    ├─→ 读取图像
    ├─→ YOLO检测
    ├─→ 物体跟踪
    ├─→ 入桶检测
    ├─→ 体积估测
    ├─→ 可视化生成
    └─→ 更新UI显示
    
[6] 生成结果
    ├─→ 保存可视化帧
    ├─→ 生成分析视频
    ├─→ 生成统计图表
    ├─→ 保存JSON结果
    └─→ 显示完成提示
    
[7] 用户查看结果
    └─→ 打开输出目录查看
```

## 🔑 关键设计决策

1. **UI框架选择**: PyQt6
   - 跨平台支持
   - 丰富的组件库
   - 良好的文档和社区支持

2. **核心逻辑复用**: 直接导入MIx模块
   - 保证算法一致性
   - 避免代码重复
   - 便于维护更新

3. **多线程设计**: QThread
   - 避免UI冻结
   - 支持异步处理
   - 信号槽通信机制

4. **配置管理**: YAML + 默认配置
   - 灵活的参数调整
   - 配置持久化
   - 多场景支持

5. **可视化策略**: 实时+离线混合
   - 实时预览帮助调试
   - 离线生成完整视频
   - 平衡性能和效果

## 📈 性能优化点

1. **图像去重**: 防止重复处理
2. **帧同步**: 避免计数偏差
3. **懒加载**: 延迟模型初始化
4. **内存管理**: 及时释放大对象
5. **批量处理**: 减少IO开销

---

**项目完成日期**: 2025年10月12日  
**最终版本**: v1.0.0  
**作者**: Sylvan  
**状态**: ✅ 完成

