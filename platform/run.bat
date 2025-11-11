@echo off
REM Construction Waste Monitor 启动脚本

echo ========================================
echo Construction Waste Monitor 桌面客户端
echo ========================================
echo.

REM 检查Python
python --version >nul 2>&1
if errorlevel 1 (
    echo 错误: 未找到Python
    echo 请先安装Python 3.8或更高版本
    pause
    exit /b 1
)

echo 正在启动程序...
echo.

python main.py

if errorlevel 1 (
    echo.
    echo 程序运行出错！
    echo 请查看日志文件: logs/construction_waste_monitor.log
    pause
)

