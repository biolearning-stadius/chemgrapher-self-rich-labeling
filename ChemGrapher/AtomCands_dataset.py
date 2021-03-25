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

class AtomCands_dataset(Dataset):
      
      def __init__(self, csv_file, dataset_dir, dataset_dir2=None, labelled=False, multiply=1, rare_atoms=None, rare_multiply=1):
          self.translate_file = pd.read_csv(csv_file,index_col='index')
          self.dataset_dir = dataset_dir
          self.labelled = labelled
          self.multiply = multiply
          self.rare_atoms = rare_atoms
          self.rare_multiply = rare_multiply
      def __len__(self):
          offset=0
          if self.rare_atoms is not None:
             offset=len(self.rare_atoms)
          return len(np.unique(self.translate_file.index.values))*self.multiply+offset*self.rare_multiply

      def __getitem__(self, idx):
          dict_atom = {'empty': '0', 'C': '1', 'H': 2, 'N': '3', 'O': '4', 'S': '5', 'F': '6', 'Cl': '7', 'Br': '8', 'I': '9', 'Se': '10', 'P': '11', 'B': '12', 'Si': '13'}
          dict_charge = {'nocharge': '0', '-1': '1',  '0': '2', '1': '3'}
          
          target_atom=np.full((1, 1), 0)
          target_charge=np.full((1, 1), 0)
          normal_len = len(np.unique(self.translate_file.index.values))
          offset=0
          if self.rare_atoms is not None:
             offset=len(self.rare_atoms)
      #       import ipdb; ipdb.set_trace()
          unique_len = normal_len+offset*self.rare_multiply
          idx = idx%unique_len
          if idx >= normal_len:
        #     import ipdb; ipdb.set_trace() 
             idx = int(self.rare_atoms[(idx-normal_len)%len(self.rare_atoms)])

          translate_df = self.translate_file.loc[idx]
          
          molid = int(translate_df['molid'])
          x = int(translate_df['atomcoord2'])
          y = int(translate_df['atomcoord1'])

          folderindex = idx // 1000
          input_tensor = torch.load(f'{self.dataset_dir}/{str(folderindex)}/atom_{str(idx)}')    
          input_tensor2 = torch.load(f'{self.dataset_dir}/{str(folderindex)}/charge_{str(idx)}')          
          
          molid_np=np.full((1, 1), 0)
          x_np=np.full((1, 1), 0)
          y_np=np.full((1, 1), 0)
          x_np[0,0] = x
          y_np[0,0] = y
          molid_np[0,0] = molid
          molid_tensor = torch.from_numpy(molid_np.astype(np.int64))
          x_tensor = torch.from_numpy(x_np.astype(np.int64))
          y_tensor = torch.from_numpy(y_np.astype(np.int64))

          if self.labelled:
             target_atom[0,0]=int(dict_atom.setdefault(str(translate_df['atom']), '14'))
             target_charge[0,0]=int(dict_charge.setdefault(str(translate_df['charge']), '4'))
             if target_atom[0,0] == int('14'):
                print(f"molid {translate_df['molid']}")
          target_atom_tensor = torch.from_numpy(target_atom.astype(np.int64))
          target_charge_tensor = torch.from_numpy(target_charge.astype(np.int64))

          sample = {'input': input_tensor, 'input2': input_tensor2, 'target_atom': target_atom_tensor, 'target_charge': target_charge_tensor, 'molid': molid_tensor, 'x': x_tensor, 'y': y_tensor} 
          
          return sample
