#!/bin/bash

# 遍历当前目录下所有的 sent{id}_frames 文件夹
for folder in sent*_frames; do
    # 如果是文件夹
    if [ -d "$folder" ]; then
        # 进入文件夹
        cd "$folder" || exit
        # 遍历文件夹内的前 8 张图片
        for ((i=0; i<8; i++)); do
            # 复制图片并重命名为 frame{i+8}.jpg
            cp "frame${i}.jpg" "frame$((i+8)).jpg"
        done
        # 退出文件夹
        cd ..
    fi
done

