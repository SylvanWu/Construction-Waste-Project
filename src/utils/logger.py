"""
日志配置模块
"""

import sys
from pathlib import Path
from loguru import logger


def setup_logger(log_dir: Path = None, verbose: bool = False):
    """
    设置日志系统
    
    Args:
        log_dir: 日志目录
        verbose: 是否显示详细日志
    """
    # 移除默认的logger
    logger.remove()
    
    # 设置日志级别
    level = "DEBUG" if verbose else "INFO"
    
    # 添加控制台输出
    logger.add(
        sys.stdout,
        level=level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
        colorize=True
    )
    
    # 如果提供了日志目录，添加文件输出
    if log_dir:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        log_file = log_dir / "mix_processing.log"
        logger.add(
            log_file,
            level=level,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {module}:{function}:{line} | {message}",
            rotation="10 MB",
            retention="7 days",
            encoding="utf-8"
        )
    
    return logger
