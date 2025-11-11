#!/bin/bash
# Construction Waste Monitor 启动脚本 (Linux/Mac)

echo "========================================"
echo "Construction Waste Monitor 桌面客户端"
echo "========================================"
echo

# 检查Python
if ! command -v python3 &> /dev/null; then
    echo "错误: 未找到Python3"
    echo "请先安装Python 3.8或更高版本"
    exit 1
fi

echo "正在启动程序..."
echo

python3 main.py

if [ $? -ne 0 ]; then
    echo
    echo "程序运行出错！"
    echo "请查看日志文件: logs/construction_waste_monitor.log"
    read -p "按Enter键退出..."
fi

