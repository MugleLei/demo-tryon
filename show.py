import argparse
import numpy as np
from PIL import Image

parser = argparse.ArgumentParser(description='show result in grid')
parser.add_argument("--src", type=str, default='')
parser.add_argument("--dst", type=str, default='')

args = parser.parse_args()

src_name = f"{args.src.split('.')[0]}.png"
dst_name = f"{args.dst.split('.')[0]}.png"

output_grid = np.concatenate([
                np.array(Image.open(f'ACGPN/Data_preprocessing/test_color/{src_name}')),
                np.array(Image.open(f'results/warped-cloth_{dst_name}')),
                np.array(Image.open(f'ACGPN/Data_preprocessing/test_img/{dst_name}')),
                np.array(Image.open(f'results/refined-cloth_{dst_name}')),
                np.array(Image.open(f'results/try-on_{dst_name}'))
                ], axis=1)
image_grid = Image.fromarray(output_grid)
image_grid.save(f"results/{src_name[:-4]}-on-{dst_name}")