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
from decimal import Decimal, ROUND_HALF_UP
from math import sqrt
from scipy.spatial.distance import pdist

class RoadLayerDataset(Dataset):
      
      def __init__(self, csv_file, last_layer_dir="/staging/leuven/stg_00001/last_layer/", image_dir="new_images/", selection=False, data_index=[0], labelled=True):
          self.last_layer = pd.read_csv(csv_file,index_col='index')
          self.last_layer_dir = last_layer_dir
          self.image_dir = image_dir
          self.selection = selection
          self.data_index = data_index
          self.labelled = labelled
          self.filter_size = 3
      
      def __len__(self):
          if self.selection:
             return len(self.data_index)
          else:
             return len(np.unique(self.last_layer.index.values))
         # molids=self.compounds[['molid']]
         # return len(molids.drop_duplicates())

      def __getitem__(self, idx):
          dict = {'0.0': '0', '1.0': '1', '2.0': '2', '3.0': '3', '4.0': '4', '4.5': '5', '5.0': '6', '5.5': '7'}
          #if idx==19:
          #    imagename='out.bmp'
          #else:
          if self.selection:
          #   folderindex = self.data_index[idx] // 1000
          #   imageindex = self.data_index[idx]
             idx=self.data_index[idx]
          #else:
          #   folderindex = idx // 1000
          #   imageindex=idx


          last_layer_2_df = self.last_layer.loc[idx]
          
          last_layer_molid = int(last_layer_2_df['molid'])
          
          folderindex = last_layer_molid // 1000
          

          layername=self.last_layer_dir+str(folderindex)+"/"+str(last_layer_molid)+'.numpy.npy'
          img_folderindex=folderindex
          imageindex=last_layer_molid

          imagename=self.image_dir+str(img_folderindex)+"/"+str(imageindex)+'.png'

          last_layer_np = np.load(layername)
 
          image = io.imread(imagename)
          image[:3,:3,:]
          image2=image==[255,255,255]
          image=np.all(image2,axis=2)
          image=1-image.astype(np.float32)
        
          target=np.full((1, 1), 0)
          cand1_as_input = np.full((image.shape[0],image.shape[0]), 0)
          molid_np=np.full((1, 1), 0)
          x1_np=np.full((1, 1), 0)
          y1_np=np.full((1, 1), 0)
          x2_np=np.full((1, 1), 0)
          y2_np=np.full((1, 1), 0)
          molid_np[0,0] = last_layer_molid
          x1_np[0,0] = int(last_layer_2_df[['cand1coord2']])
          y1_np[0,0] = int(last_layer_2_df[['cand1coord1']])
          x2_np[0,0] = int(last_layer_2_df[['cand2coord2']])
          y2_np[0,0] = int(last_layer_2_df[['cand2coord1']])
         # numrows=np.size(atoms,0)
         # i=0
          
          basepoint_x=int(last_layer_2_df[['cand1coord2']])
          basepoint_y=int(last_layer_2_df[['cand1coord1']])
          endpoint_x=int(last_layer_2_df[['cand2coord2']])
          endpoint_y=int(last_layer_2_df[['cand2coord1']])


          rr, cc = draw.line(basepoint_x, basepoint_y, endpoint_x, endpoint_y)

          cand1_as_input[rr, cc]=1
           
          if self.labelled:
             target[0,0]=int(dict[str(last_layer_2_df['bondtype'])])
           
          cand1_as_input=ndimage.maximum_filter(cand1_as_input,size=self.filter_size)
#          target = ndimage.maximum_filter(target,size=10)
          last_layer_tensor = torch.from_numpy(last_layer_np.astype(np.float32))
#          import ipdb; ipdb.set_trace()
          #bondlayer_tensor=last_layer_tensor.index_select(0,torch.LongTensor((16,17,18,19,20,21,22,23)))
          bondlayer_tensor=last_layer_tensor.index_select(0,torch.LongTensor((3,4,5)))
          imagetensor = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
          cand1_as_input_tensor = torch.from_numpy(cand1_as_input.astype(np.float32)).unsqueeze(0)
          input_tensor = torch.cat((imagetensor,bondlayer_tensor,cand1_as_input_tensor),0)
          #imagetensor = torch.from_numpy(image.astype(np.float32))
          targettensor = torch.from_numpy(target.astype(np.int64))
          molid_tensor = torch.from_numpy(molid_np.astype(np.int64))
          x1_tensor = torch.from_numpy(x1_np.astype(np.int64))
          y1_tensor = torch.from_numpy(y1_np.astype(np.int64))
          x2_tensor = torch.from_numpy(x2_np.astype(np.int64))
          y2_tensor = torch.from_numpy(y2_np.astype(np.int64))
          sample = {'input': input_tensor, 'target': targettensor, 'molid': molid_tensor, 'x1': x1_tensor, 'y1': y1_tensor, 'x2': x2_tensor, 'y2': y2_tensor} 
          #sample = {imagetensor, targettensor} 
          
 
          return sample
