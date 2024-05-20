#!/bin/bash

# 遍历当前目录下所有形如"sample_*.mp4"的文件
for file in sample_*.json; do
    # 使用正则表达式提取文件中的数字部分
    if [[ $file =~ sample_([0-9]+)\.json ]]; then
        num=${BASH_REMATCH[1]}
        # 重命名文件
        mv "$file" "$num.json"
    fi
done

