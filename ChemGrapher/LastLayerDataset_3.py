from torch.utils.data import *
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

class LastLayerDataset_3(Dataset):
      
      def __init__(self, csv_file, last_layer_dir="/staging/leuven/stg_00001/last_layer/", image_dir="new_images/", selection=False, data_index=[0]):
          self.last_layer = pd.read_csv(csv_file,index_col='index')
          self.last_layer_dir = last_layer_dir
          self.image_dir = image_dir
          self.selection = selection
          self.data_index = data_index
      
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
      #    image=cv2.imread(imagename)
          if self.image_dir == "new_images/":
             img_folderindex=folderindex+90
             imageindex=last_layer_molid+90000
          else:
             img_folderindex=folderindex
             imageindex=last_layer_molid

          imagename=self.image_dir+str(img_folderindex)+"/"+str(imageindex)+'.png'

          last_layer_np = np.load(layername)
 
          image = io.imread(imagename)
          image[:3,:3,:]
          image2=image==[255,255,255]
          image=np.all(image2,axis=2)
          image=1-image.astype(np.float32)
        
         

    #      cand1_df=last_layer_2_df[['cand1coord1','cand1coord2']]
    #      cand2_df=last_layer_2_df[['cand2coord1','cand2coord2']]


         # atoms=atomsdf.as_matrix()
        
#          import ipdb; ipdb.set_trace()
         # target=np.zeros((1000, 1000))
          target=np.full((1, 1), 0)
          cand1_as_input = np.full((1000,1000), 0)
          cand2_as_input = np.full((1000,1000), 0)

         # numrows=np.size(atoms,0)
         # i=0
          centerx=int((int(last_layer_2_df[['cand1coord2']])+int(last_layer_2_df[['cand2coord2']]))/2)
          centery=int((int(last_layer_2_df[['cand1coord1']])+int(last_layer_2_df[['cand2coord1']]))/2)
          beginx=centerx-62
          endx=centerx+62
          beginy=centery-62
          endy=centery+62
          
          basepoint_x=int(last_layer_2_df[['cand1coord2']])
          basepoint_y=int(last_layer_2_df[['cand1coord1']])
          endpoint_x=int(last_layer_2_df[['cand2coord2']])
          endpoint_y=int(last_layer_2_df[['cand2coord1']])

          centerx=int((basepoint_x+endpoint_x)/2)
          centery=int((basepoint_y+endpoint_y)/2)

          #move line to source vector
          vector_x = endpoint_x-basepoint_x
          vector_y = endpoint_y-basepoint_y

          vector_length = sqrt(vector_x*vector_x + vector_y*vector_y)
          v_x = vector_x/vector_length
          v_y = vector_y/vector_length
          l5_v_x = v_x*5
          l5_v_y = v_y*5


          #create two centerpoints separated with 10 pixels

          centerx_basepoint =int(centerx - l5_v_x)
          centery_basepoint =int(centery - l5_v_y)

          centerx_endpoint =int(centerx + l5_v_x)
          centery_endpoint =int(centery + l5_v_y)

          rr, cc = draw.line(basepoint_x, basepoint_y, centerx_basepoint, centery_basepoint)
         #           target[rr, cc]=int(dict[str(float(row['bondtype'])+diff_add)])

         # cand1_as_input[int(last_layer_2_df[['cand1coord2']]),int(last_layer_2_df[['cand1coord1']])]=1
          cand1_as_input[rr, cc]=1
         # cand2_as_input[int(last_layer_2_df[['cand2coord2']]),int(last_layer_2_df[['cand2coord1']])]=1
           
          rr, cc = draw.line(endpoint_x, endpoint_y, centerx_endpoint, centery_endpoint)
          cand2_as_input[rr, cc]=1
          target[0,0]=int(dict[str(last_layer_2_df['bondtype'])])
           
          image=image[np.ix_(np.arange(beginx,endx),np.arange(beginy,endy))] 
          cand1_as_input=cand1_as_input[np.ix_(np.arange(beginx,endx),np.arange(beginy,endy))]
          cand1_as_input=ndimage.maximum_filter(cand1_as_input,size=10)
          cand2_as_input=cand2_as_input[np.ix_(np.arange(beginx,endx),np.arange(beginy,endy))]
          cand2_as_input=ndimage.maximum_filter(cand2_as_input,size=10)
          last_layer_np=last_layer_np[np.ix_(np.arange(0,last_layer_np.shape[0]),np.arange(beginx,endx),np.arange(beginy,endy))]
#          target = ndimage.maximum_filter(target,size=10)
          last_layer_tensor = torch.from_numpy(last_layer_np.astype(np.float32))
          imagetensor = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
          cand1_as_input_tensor = torch.from_numpy(cand1_as_input.astype(np.float32)).unsqueeze(0)
          cand2_as_input_tensor = torch.from_numpy(cand2_as_input.astype(np.float32)).unsqueeze(0)
          input_tensor = torch.cat((imagetensor,last_layer_tensor,cand1_as_input_tensor,cand2_as_input_tensor),0)
          #imagetensor = torch.from_numpy(image.astype(np.float32))
          targettensor = torch.from_numpy(target.astype(np.int64))
          sample = {'input': input_tensor, 'target': targettensor} 
          #sample = {imagetensor, targettensor} 
          
 
          return sample
