"""
Construction Waste Monitor 桌面客户端 - 主入口
建筑垃圾监测与体积估测桌面应用程序
"""

import sys
import os
from pathlib import Path

# 添加项目路径
ROOT_DIR = Path(__file__).parent
sys.path.insert(0, str(ROOT_DIR))

# 启用Tracker调试模式
os.environ['TRACKER_DEBUG'] = '1'

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QIcon
from app.main_window import MainWindow
from loguru import logger


def setup_logger():
    """配置日志系统"""
    log_dir = ROOT_DIR / "logs"
    log_dir.mkdir(exist_ok=True)
    
    log_file = log_dir / "construction_waste_monitor.log"
    
    # 配置loguru - 启用DEBUG级别以查看详细跟踪信息
    logger.add(
        log_file,
        rotation="10 MB",
        retention="7 days",
        level="DEBUG",
        encoding="utf-8"
    )
    
    logger.info("=" * 60)
    logger.info("Construction Waste Monitor 桌面客户端启动")
    logger.info("=" * 60)


def main():
    """主函数"""
    try:
        # 配置日志
        setup_logger()
        
        # 创建应用
        app = QApplication(sys.argv)
        app.setApplicationName("Construction Waste Monitor")
        app.setOrganizationName("Sylvan")
        app.setApplicationVersion("1.0.0")
        
        # 加载样式
        style_file = ROOT_DIR / "resources" / "styles" / "main.qss"
        if style_file.exists():
            with open(style_file, 'r', encoding='utf-8') as f:
                app.setStyleSheet(f.read())
        
        # 创建并显示主窗口
        window = MainWindow()
        window.show()
        
        logger.info("主窗口已显示")
        
        # 运行应用
        sys.exit(app.exec())
        
    except Exception as e:
        logger.error(f"应用启动失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()

