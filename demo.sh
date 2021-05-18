#!/usr/bin/env bash

echo "衣服模特图片：$1"
echo "试穿模特图片：$2"

dp_dir=ACGPN/Data_preprocessing

rm ${dp_dir}/test_color/*
rm ${dp_dir}/test_edge/*
rm ${dp_dir}/test_img/*
rm ${dp_dir}/test_label/*
rm ${dp_dir}/test_pose/*
# rm ACGPN/Data_preprocessing/test_colormask/*
# rm ${dp_dir}/test_mask/*

# 在test_pairs中记录配对信息
# 将衣服模特图片放在test_color中
# 将试穿模特图片放在test_img中
# 计算试穿模特的姿势，将结果保存在test_pose中
python demo.py --src $1 --dst $2

# 将衣服模特mask放在test_edge中
python schp/upcloth_extractor.py --input-dir "${dp_dir}/test_color" --output-dir "${dp_dir}/test_edge"

# 对test_img中试穿模特的衣服进行分割
# 试穿模特衣服mask保存在test_label中
python schp/simple_extractor.py --input-dir "${dp_dir}/test_img" --output-dir "${dp_dir}/test_label"

# 综合衣服模特图片、衣服模特mask、试穿模特图片、试穿模特mask和模特pose进行生成
python ACGPN/test.py

python show.py --src $1 --dst $2

echo "demo finished."