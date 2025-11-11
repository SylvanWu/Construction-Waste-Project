"""
安装脚本
使用setuptools创建可安装的包
"""

from setuptools import setup, find_packages
from pathlib import Path

# 读取README
readme_file = Path(__file__).parent.parent / "README.md"
long_description = ""
if readme_file.exists():
    with open(readme_file, 'r', encoding='utf-8') as f:
        long_description = f.read()

# 读取依赖
requirements_file = Path(__file__).parent.parent / "requirements.txt"
requirements = []
if requirements_file.exists():
    with open(requirements_file, 'r', encoding='utf-8') as f:
        requirements = [
            line.strip() 
            for line in f 
            if line.strip() and not line.strip().startswith('#')
        ]

setup(
    name="construction-waste-monitor",
    version="1.0.0",
    author="Sylvan",
    author_email="",
    description="建筑垃圾监测与体积估测桌面应用程序",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: Microsoft :: Windows",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        'console_scripts': [
            'construction-waste-monitor=main:main',
        ],
    },
    include_package_data=True,
    package_data={
        '': ['resources/**/*'],
    },
)

