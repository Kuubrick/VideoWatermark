#!/bin/bash

# 遍历当前目录下的所有子文件夹
for dir in sent*/; do
    # 去除目录路径后面的'/'
    dir_name=$(basename "$dir" /)
    
    # 重命名文件夹，添加"_frames"后缀
    new_name="${dir_name}_frames"
    
    # 重命名文件夹
    mv "$dir" "$new_name"
done

echo "重命名完成。"

