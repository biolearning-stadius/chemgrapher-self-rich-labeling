from torch.utils.data import Dataset
import pandas as pd
import skimage.io as io
import skimage.draw as draw
import numpy as np
from scipy import ndimage, misc
from skimage.color import rgb2gray
from matplotlib import pyplot as plt
import cv2
import torch
import random
from decimal import Decimal, ROUND_HALF_UP
from math import sqrt
from scipy.spatial.distance import pdist

class BondCands_dataset(Dataset):
      
      def __init__(self, csv_file, dataset_dir):
          self.translate_file = pd.read_csv(csv_file,index_col='index')
          self.dataset_dir = dataset_dir
      
      def __len__(self):
          return len(np.unique(self.translate_file.index.values))

      def __getitem__(self, idx):
          translate_df = self.translate_file.loc[idx]
          
          molid = int(translate_df['molid'])
          x1 = int(translate_df[['cand1coord2']])
          y1 = int(translate_df[['cand1coord1']])
          x2 = int(translate_df[['cand2coord2']])
          y2 = int(translate_df[['cand2coord1']])

          folderindex = idx // 1000
          input_tensor = torch.load(f'{self.dataset_dir}/{str(folderindex)}/bond_{str(idx)}')    
          
          molid_np=np.full((1, 1), 0)
          x1_np=np.full((1, 1), 0)
          y1_np=np.full((1, 1), 0)
          x1_np[0,0] = x1
          y1_np[0,0] = y1
          x2_np=np.full((1, 1), 0)
          y2_np=np.full((1, 1), 0)
          x2_np[0,0] = x2
          y2_np[0,0] = y2
          molid_np[0,0] = molid
          molid_tensor = torch.from_numpy(molid_np.astype(np.int64))
          x1_tensor = torch.from_numpy(x1_np.astype(np.int64))
          y1_tensor = torch.from_numpy(y1_np.astype(np.int64))
          x2_tensor = torch.from_numpy(x2_np.astype(np.int64))
          y2_tensor = torch.from_numpy(y2_np.astype(np.int64))
          sample = {'input': input_tensor, 'molid': molid_tensor, 'x1': x1_tensor, 'y1': y1_tensor, 'x2': x2_tensor, 'y2': y2_tensor} 
          
          return sample
