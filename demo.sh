#!/usr/bin/env bash

echo "试穿衣服：$1"
echo "试穿模特：$2"

dp_dir=ACGPN/Data_preprocessing

rm ${dp_dir}/test_color/*
rm ${dp_dir}/test_edge/*
rm ${dp_dir}/test_img/*
rm ${dp_dir}/test_label/*
rm ${dp_dir}/test_pose/*
# rm ACGPN/Data_preprocessing/test_colormask/*
# rm ${dp_dir}/test_mask/*

# 将衣服图片放在test_color中
# 将衣服mask放在test_edge中
# 将试穿模特放在test_img中
# 计算试穿模特的姿势，将结果保存在test_pose中
python demo.py --src $1 --dst $2

# 对test_img中试穿模特的衣服进行分割
# 试穿模特衣服mask保存在test_label中
python schp/simple_extractor.py --input-dir "${dp_dir}/test_img" --output-dir "${dp_dir}/test_label"

# 综合衣服图片、衣服mask、模特图片、模特mask和模特pose进行生成
python ACGPN/test.py

echo "demo finished."