#!/bin/bash

# 获取所有符合模式的文件
files=(`ls *_w.json 2>/dev/null`)

# 按照pid排序
sorted_files=($(printf "%s\n" "${files[@]}"|sort -t_ -k1n))

counter=0

# 遍历排序后的文件
for file in "${sorted_files[@]}"; do
  # 从文件名中提取pid和name部分
  pid=${file%%_*}
  name=${file%_w.json}

  # 重命名文件
  mv "$file" "$counter""_w.json"
  mv "${name}.mp4" "$counter.mp4"
  mv "${name}_w.mp4" "$counter""_w.mp4"

  # 计数器增加
  ((counter++))
done
