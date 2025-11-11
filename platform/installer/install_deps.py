"""
依赖安装工具
用于首次运行时安装必要的依赖包
"""

import subprocess
import sys
from pathlib import Path


def install_package(package: str) -> bool:
    """
    安装Python包
    
    Args:
        package: 包名
        
    Returns:
        是否安装成功
    """
    try:
        print(f"正在安装 {package}...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", package],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        print(f"✓ {package} 安装成功")
        return True
    except Exception as e:
        print(f"✗ {package} 安装失败: {e}")
        return False


def check_and_install_deps():
    """检查并安装所有依赖"""
    requirements_file = Path(__file__).parent.parent / "requirements.txt"
    
    if not requirements_file.exists():
        print("错误: 未找到 requirements.txt")
        return False
    
    print("=" * 60)
    print("Construction Waste Monitor 依赖安装工具")
    print("=" * 60)
    print()
    
    # 读取依赖列表
    with open(requirements_file, 'r', encoding='utf-8') as f:
        packages = [
            line.strip() 
            for line in f 
            if line.strip() and not line.strip().startswith('#')
        ]
    
    print(f"需要安装 {len(packages)} 个包...")
    print()
    
    # 安装每个包
    success_count = 0
    for package in packages:
        if install_package(package):
            success_count += 1
    
    print()
    print("=" * 60)
    print(f"完成! 成功安装 {success_count}/{len(packages)} 个包")
    print("=" * 60)
    
    return success_count == len(packages)


if __name__ == "__main__":
    check_and_install_deps()
    input("\n按Enter键退出...")

