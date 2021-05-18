# import cv2
import argparse
# import numpy as np
from PIL import Image
import os

from ACGPN.predict_pose import generate_pose_keypoints
# from U2Net import u2net_load
# from U2Net import u2net_run

parser = argparse.ArgumentParser(description='try on demo')
parser.add_argument("--src", type=str, default='')
parser.add_argument("--dst", type=str, default='')

args = parser.parse_args()
print(f"intput src: {args.src}, dst: {args.dst}")

src_name = f"{args.src.split('.')[0]}.png"
dst_name = f"{args.dst.split('.')[0]}.png"

with open('ACGPN/Data_preprocessing/test_pairs.txt','w') as f:
    f.write(f'{dst_name} {src_name}')

cloth = Image.open(args.src)
cloth = cloth.resize((192, 256), Image.BICUBIC).convert('RGB')
cloth.save(os.path.join('ACGPN/Data_preprocessing/test_color', src_name))

# u2net = u2net_load.model(model_name = 'u2netp')
# u2net_run.infer(u2net, 'ACGPN/Data_preprocessing/test_color', 'ACGPN/Data_preprocessing/test_edge')

img = Image.open(args.dst)
img = img.resize((192,256), Image.BICUBIC)

img_path = os.path.join('ACGPN/Data_preprocessing/test_img', dst_name)
img.save(img_path)

img_path = os.path.join('ACGPN/Data_preprocessing/test_img', dst_name)
pose_path = os.path.join('ACGPN/Data_preprocessing/test_pose', dst_name[:-4]+'_keypoints.json')
generate_pose_keypoints(img_path, pose_path)


# print("demo finished.")