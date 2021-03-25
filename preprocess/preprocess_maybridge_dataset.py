import cv2                                                                        
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt
import os
import csv
from rdkit import Chem
import argparse
from itertools import islice

parser = argparse.ArgumentParser(description="Preprocess Maybridge Dataset for ChemGrapher")
parser.add_argument("--dataset_folder", help="Dataset folder with Maybridge images", type=str, required=True)
parser.add_argument("--output_folder", help="Folder to save preprocessed images", type=str, required=True)
parser.add_argument("--num_images", help="Number of images to preprocess", type=int, default=10)
args = parser.parse_args()

file = open(f"{args.output_folder}/smiles2.csv","w")
fieldnames = ['molid','smiles']
writer = csv.DictWriter(file, fieldnames=fieldnames)
writer.writeheader()

img_files = [name for name in os.listdir(f"{args.dataset_folder}/tif/") if os.path.isfile(f"{args.dataset_folder}/tif/{name}")]
img_files.sort()
#for i in range(50000):
#img = Image.open(f"../Downloads/uob_sdag_molecule_dataset/tif/maybridge-0001-31314370.tif")

for i,img_file in enumerate(islice(img_files,args.num_images)):
    img = cv2.imread(f"{args.dataset_folder}/tif/{img_file}")
    split_name = img_file.split('.')
    mol = Chem.MolFromMolFile(f"{args.dataset_folder}/mol/{split_name[0]}.mol")
    molsmiles=Chem.MolToSmiles(mol)
    writer.writerow({'molid': i, 'smiles': molsmiles})
    img_scaled = cv2.resize(img,None,fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC) 
    folder_index = i//1000
    dirName = f"{args.output_folder}/{folder_index}"
    if not os.path.exists(dirName):
            os.makedirs(dirName)
            print("Directory " , dirName ,  " Created ")
#ret,th1 = cv2.threshold(img_scaled,200,255,cv2.THRESH_BINARY)
    cv2.imwrite(f"{dirName}/{i}.png", img_scaled)
    image1 = Image.open(f"{dirName}/{i}.png")
    new_im = Image.new('RGB', (1000, 1000))
    new_im.paste( (255,255,255), [0,0,new_im.size[0],new_im.size[1]])
    x_offset = 150 
    y_offset = 150
    new_im.paste(image1, (x_offset,y_offset))
#imshow(np.asarray(new_im))
    new_im.save(f"{dirName}/{i}.png")

file.close()
