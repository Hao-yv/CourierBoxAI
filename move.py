"""
@Time    : 2025/11/11 11:28
@Author  : Zhang Haoyu
@File    : move.py
@IDE     : PyCharm
"""
import os
import shutil
from pathlib import Path


def classify_and_move_files(source_dir=".", temp_dir="temp", image_dir="image", pkl_dir=".pkl"):
    """
    对源目录中的文件进行分类处理：
    - 临时文件（扩展名以 .tmp, .temp, .log 等结尾）移动到 temp/ 文件夹
    - 图片文件（扩展名 .jpg, .jpeg, .png, .gif, .bmp 等）移动到 image/ 文件夹
    - 其他文件保持原位或根据需要扩展分类（这里仅示例两种）

    Args:
        source_dir (str): 源目录路径，默认当前目录 "."
        temp_dir (str): 临时文件目标目录
        image_dir (str): 图片文件目标目录
    """
    # 创建目标目录如果不存在
    Path(temp_dir).mkdir(exist_ok=True)
    Path(image_dir).mkdir(exist_ok=True)
    Path(pkl_dir).mkdir(exist_ok=True)

    # 定义文件类型映射（可扩展）
    temp_extensions = {'.tmp', '.temp', '.log', '.cache', '.bak'}  # 临时文件扩展名
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.svg', '.webp'}  # 图片扩展名
    pkl_extensions = {'..pkl', '..pkl'}

    # 遍历源目录中的所有文件
    for file_path in Path(source_dir).iterdir():
        if file_path.is_file():  # 只处理文件
            file_ext = file_path.suffix.lower()

            if file_ext in temp_extensions:
                # 移动到 temp/
                dest_path = Path(temp_dir) / file_path.name
                shutil.move(str(file_path), str(dest_path))
                print(f"临时文件移动: {file_path.name} -> {temp_dir}/")

            elif file_ext in image_extensions:
                # 移动到 image/
                dest_path = Path(image_dir) / file_path.name
                shutil.move(str(file_path), str(dest_path))
                print(f"图片文件移动: {file_path.name} -> {image_dir}/")

            elif file_ext in pkl_extensions:
                dest_path = Path(pkl_dir) / file_path.name
                shutil.move(str(file_path), str(dest_path))
                print(f"<UNK>: {file_path.name} -> {pkl_dir}/")

            else:
                # 其他文件：这里可以扩展更多分类，例如文档到 docs/ 等
                print(f"其他文件 {file_path.name} 保持原位（可扩展分类）")

    print("文件分类处理完成！")


if __name__ == "__main__":
    classify_and_move_files()