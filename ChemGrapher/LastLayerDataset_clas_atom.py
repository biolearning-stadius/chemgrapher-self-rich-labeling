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

class LastLayerDataset_clas_atom(Dataset):
      
      def __init__(self, csv_file, last_layer_dir="/staging/leuven/stg_00001/last_layer/", image_dir="new_images/", selection=False, data_index=[0], reduced=False, charge=False, labelled=True, filter_size=11, big_filter_size=101, full_size=1000, multiply=1):
 #         self.last_layer = pd.read_csv(csv_file,index_col='index')
          self.last_layer = pd.read_csv(csv_file)
          self.last_layer_dir = last_layer_dir
          self.image_dir = image_dir
          self.selection = selection
          self.data_index = data_index
          self.reduced = reduced
          self.charge = charge
          self.labelled = labelled
          self.filter_size = filter_size
          self.big_filter_size = big_filter_size
          self.full_size = full_size
          self.multiply = multiply
      
      def shuffle(self):
          groups = [df for _, df in self.last_layer.groupby('molid')]

          random.shuffle(groups)

          shuffled = pd.concat(groups).reset_index(drop=True)
         # import ipdb; ipdb.set_trace()

          self.last_layer = shuffled
          
      def __len__(self):
          if self.selection:
             return len(self.data_index)
          else:
             return len(np.unique(self.last_layer.index.values))*self.multiply
         # molids=self.compounds[['molid']]
         # return len(molids.drop_duplicates())

      def __getitem__(self, idx):
          dict_atom = {'empty': '0', 'C': '1', 'H': 2, 'N': '3', 'O': '4', 'S': '5', 'F': '6', 'Cl': '7', 'Br': '8', 'I': '9', 'Se': '10', 'P': '11', 'B': '12', 'Si': '13'}
          dict_charge = {'nocharge': '0', '-1': '1',  '0': '2', '1': '3'}
          #if idx==19:
          #    imagename='out.bmp'
          #else:
          if self.selection:
          #   folderindex = self.data_index[idx] // 1000
          #   imageindex = self.data_index[idx]
             idx=self.data_index[idx]
          else:
             unique_len = len(np.unique(self.last_layer.index.values))
             idx = idx%unique_len
          #else:
          #   folderindex = idx // 1000
          #   imageindex=idx

          
          last_layer_2_df = self.last_layer.loc[idx]
          
          last_layer_molid = int(last_layer_2_df['molid'])
#          last_layer_2_df = self.last_leyer.loc[last_layer_molid]

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
          target_atom=np.full((1, 1), 0)
          target_charge=np.full((1, 1), 0)
          molid_np=np.full((1, 1), 0)
          x_np=np.full((1, 1), 0)
          y_np=np.full((1, 1), 0)
          molid_np[0,0] = last_layer_molid
          cand1_as_input = np.full((self.full_size,self.full_size), 0)


#target = torch.zeros(30, 35, 512)
#source = torch.ones(30, 35, 49)
#target[:, :, :49] = source
         # numrows=np.size(atoms,0)
         # i=0
         # randx = random.randint(0, 3)
        #  randy = random.randint(0, 3)
          centerx=int(int(last_layer_2_df[['atomcoord2']]))
          x_np[0,0] = centerx
        #  centerx=int(int(last_layer_2_df[['atomcoord2']]+randx))
          centery=int(int(last_layer_2_df[['atomcoord1']]))
          y_np[0,0] = centery
      #    centery=int(int(last_layer_2_df[['atomcoord1']]+randy))
          if self.big_filter_size < self.full_size:
             big_cut = int((self.big_filter_size - 1)/2)
             begin_big_x=centerx-big_cut
             end_big_x=centerx+big_cut
             begin_big_y=centery-big_cut
             end_big_y=centery+big_cut
          else:
             begin_big_x = 0
             end_big_x = self.full_size-1
             begin_big_y = 0
             end_big_y = self.full_size-1
          
          small_cut = int((self.filter_size - 1)/2)
          begin_small_x=centerx-small_cut
          end_small_x=centerx+small_cut
          begin_small_y=centery-small_cut
          end_small_y=centery+small_cut

          rr, cc = draw.rectangle((begin_small_x,begin_small_y), extent=(self.filter_size,self.filter_size))
         #           target[rr, cc]=int(dict[str(float(row['bondtype'])+diff_add)])
          #import ipdb; ipdb.set_trace()
          cand1_as_input[rr, cc]=1
          
          if self.labelled:
             target_atom[0,0]=int(dict_atom.setdefault(str(last_layer_2_df['atom']), '14'))
             target_charge[0,0]=int(dict_charge.setdefault(str(last_layer_2_df['charge']), '4'))
           
          image=image[np.ix_(np.arange(begin_big_x,end_big_x),np.arange(begin_big_y,end_big_y))] 
          cand1_as_input=cand1_as_input[np.ix_(np.arange(begin_big_x,end_big_x),np.arange(begin_big_y,end_big_y))]
          last_layer_np=last_layer_np[np.ix_(np.arange(0,last_layer_np.shape[0]),np.arange(begin_big_x,end_big_x),np.arange(begin_big_y,end_big_y))]
#          target = ndimage.maximum_filter(target,size=10)
          last_layer_tensor = torch.from_numpy(last_layer_np.astype(np.float32))

          if self.full_size == 1000:
             atomlayer_tensor=last_layer_tensor.index_select(0,torch.LongTensor((0,1,2,3,4,5,6,7,8,9,10,11,12,13,14)))
             chargelayer_tensor=last_layer_tensor.index_select(0,torch.LongTensor((23,24,25,26,27)))
          else:
             atomlayer_tensor=last_layer_tensor.index_select(0,torch.LongTensor((0,1,2)))
             chargelayer_tensor=last_layer_tensor.index_select(0,torch.LongTensor((6,7,8)))

          imagetensor = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
          cand1_as_input_tensor = torch.from_numpy(cand1_as_input.astype(np.float32)).unsqueeze(0)
          
          if self.reduced: 
             input_tensor = torch.cat((imagetensor,atomlayer_tensor,cand1_as_input_tensor),0)
             input_charge_tensor = torch.cat((imagetensor,chargelayer_tensor,cand1_as_input_tensor),0)
#             import ipdb; ipdb.set_trace()
          elif self.charge:
             input_tensor = torch.cat((imagetensor,chargelayer_tensor,cand1_as_input_tensor),0)
          else:
             input_tensor = torch.cat((imagetensor,last_layer_tensor,cand1_as_input_tensor),0)
          #imagetensor = torch.from_numpy(image.astype(np.float32))
          target_atom_tensor = torch.from_numpy(target_atom.astype(np.int64))
          target_charge_tensor = torch.from_numpy(target_charge.astype(np.int64))
          molid_tensor = torch.from_numpy(molid_np.astype(np.int64))
          x_tensor = torch.from_numpy(x_np.astype(np.int64))
          y_tensor = torch.from_numpy(y_np.astype(np.int64))
          if self.charge:
             sample = {'input': input_tensor, 'target_atom': target_atom_tensor, 'target_charge': target_charge_tensor, 'molid': molid_tensor, 'x': x_tensor, 'y': y_tensor}
          else:
             sample = {'input': input_tensor, 'input_charge': input_charge_tensor, 'target_atom': target_atom_tensor, 'target_charge': target_charge_tensor, 'molid': molid_tensor, 'x': x_tensor, 'y': y_tensor} 
          #sample = {imagetensor, targettensor} 
          
 
          return sample

      def get_input_tensor(last_layer_np, image, x, y):
          cand1_as_input = np.full((1000,1000), 0)
          begin_big_x=x-50
          end_big_x=x+50
          begin_big_y=y-50
          end_big_y=y+50

          begin_small_x=x-5
          end_small_x=x+5
          begin_small_y=y-5
          end_small_y=y+5

          rr, cc = draw.rectangle((begin_small_x,begin_small_y), extent=(11,11))
          cand1_as_input[rr, cc]=1
          cand1_as_input=cand1_as_input[np.ix_(np.arange(begin_big_x,end_big_x),np.arange(begin_big_y,end_big_y))]

          last_layer_np_cut=last_layer_np[np.ix_(np.arange(0,last_layer_np.shape[0]),np.arange(begin_big_x,end_big_x),np.arange(begin_big_y,end_big_y))]
          image_cut=image[np.ix_(np.arange(begin_big_x,end_big_x),np.arange(begin_big_y,end_big_y))]
          last_layer_tensor = torch.from_numpy(last_layer_np_cut.astype(np.float32))
          imagetensor = torch.from_numpy(image_cut.astype(np.float32)).unsqueeze(0)
          cand1_as_input_tensor = torch.from_numpy(cand1_as_input.astype(np.float32)).unsqueeze(0)
          input_tensor2 = torch.cat((imagetensor,last_layer_tensor,cand1_as_input_tensor),0)
          
          return input_tensor2
