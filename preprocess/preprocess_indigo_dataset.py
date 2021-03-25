import cv2                                                                        
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import re

parser = argparse.ArgumentParser(description="Preprocess Indigo Dataset for ChemGrapher")
parser.add_argument("--dataset_folder", help="Dataset folder with Indigo images", type=str, required=True)
parser.add_argument("--output_folder", help="Folder to save preprocessed images", type=str, required=True)
parser.add_argument("--num_images", help="Number of images to preprocess", type=int, default=10)
args = parser.parse_args()
pd.options.mode.chained_assignment = None 
df = pd.read_csv(f"{args.dataset_folder}/../smiles.txt", names=['molid', 'smiles'], header=None)
new_df = df[:args.num_images]
new_df['smiles'].replace(to_replace='R.',value='*', inplace=True, regex=True)
new_df.to_csv(f"{args.output_folder}/smiles.csv", index=False)
for i in range(args.num_images):
    img = cv2.imread(f"{args.dataset_folder}/{i}.png") 
    img_scaled = cv2.resize(img,None,fx=2, fy=2, interpolation = cv2.INTER_CUBIC) 

    ret,th1 = cv2.threshold(img_scaled,200,255,cv2.THRESH_BINARY)
    cv2.imwrite(f"{args.output_folder}/{i}.png", th1)
    image1 = Image.open(f"{args.output_folder}/{i}.png")
    new_im = Image.new('RGB', (1000, 1000))
    new_im.paste( (255,255,255), [0,0,new_im.size[0],new_im.size[1]])
    x_offset = 250 
    y_offset = 250
    new_im.paste(image1, (x_offset,y_offset))
#imshow(np.asarray(new_im))
    new_im.save(f"{args.output_folder}/{i}.png")
