#!/bin/bash

# 搜索所有符合模式的文件并进行重命名
for file in *_w.mp4; do
    # 提取PID（假设PID是文件名中"_"前的部分）
    pid=$(echo "$file" | cut -d '_' -f1)
    
    # 重命名文件，去掉"_w"
    mv "$file" "${pid}.mp4"
done

echo "重命名完成。"

