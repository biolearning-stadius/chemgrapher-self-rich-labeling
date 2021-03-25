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

class CompoundDataset_notriangle(Dataset):
      
      def __init__(self, csv_file, testdataset=False, image_dir="new_images/", selection=False, data_index=[0], with_edges=False, debug=False, filter_size=10, multiply=1):
          self.compounds = pd.read_csv(csv_file,index_col='molid')
          self.testdataset = testdataset
          self.image_dir = image_dir
          self.selection = selection
          self.data_index = data_index
          self.with_edges = with_edges
          self.debug = debug
          self.molids = np.unique(self.compounds.index.values)
          self.dict_bond = {'1.0': '1', '2.0': '2', '3.0': '3', '4.0': '4', '4.5': '5', '5.0': '6', '5.5': '7'}
          self.dict_atom = {'C': '1', 'H': 2, 'N': '3', 'O': '4', 'S': '5', 'F': '6', 'Cl': '7', 'Br': '8', 'I': '9', 'Se': '10', 'P': '11', 'B': '12', 'Si': '13'}
          self.dict_charge = {-1: '1',  0: '2', 1: '3'}
          self.atom_list = []
          self.bond_list = []
          self.filter_size = filter_size
          self.multiply = multiply

      def __len__(self):
          if self.selection:
             return len(self.data_index)
          else:
             return len(np.unique(self.compounds.index.values))*self.multiply
         # molids=self.compounds[['molid']]
         # return len(molids.drop_duplicates())

      def create_atom_list(self,atoms):
          atom_list = []

          for atom in atoms:
              atom_list.append([int(atom[3]),int(atom[2]),int(self.dict_atom.setdefault(atom[0], '14')),int(self.dict_charge.setdefault(atom[1], '4'))])
          
          self.atom_list = atom_list

      def create_bond_list(self,edgesdf):
          bond_list = []

          for index, row in edgesdf.iterrows():
              bond_list.append([[[int(row['atom1coord2']),int(row['atom1coord1']),row['atom1'],row['charge1']]],[[int(row['atom2coord2']),int(row['atom2coord1']),row['atom2'],row['charge2']]],int(self.dict_bond[str(row['bondtype'])])])

          self.bond_list = bond_list

      def __getitem__(self, idx):
#          dict = {'C': '1', 'H': 2, 'N': '3', 'O': '4', 'S': '5', 'F': '6', 'Cl': '7', 'Br': '8', 'I': '9', 'Se': '10', 'P': '11', 'B': '12', 'Si': '13', '1.0': '14', '2.0': '15', '3.0': '16', '4.0': '17', '4.5': '18', '5.0': '19', '5.5': '20', 'V': '21'}
          dict_atom = {'C': '1', 'H': 2, 'N': '3', 'O': '4', 'S': '5', 'F': '6', 'Cl': '7', 'Br': '8', 'I': '9', 'Se': '10', 'P': '11', 'B': '12', 'Si': '13'}
          dict_bond = {'1.0': '1', '2.0': '2', '3.0': '3', '4.0': '4', '4.5': '5', '5.0': '6', '5.5': '7'}
          dict_charge = {-1: '1',  0: '2', 1: '3'}
          #if idx==19:
          #    imagename='out.bmp'
          #else:
          if self.selection:
             folderindex = self.data_index[idx] // 1000
             imageindex = self.data_index[idx]
             idx=self.data_index[idx] 
          else:
             unique_len = len(np.unique(self.compounds.index.values))
             idx = self.molids[idx%unique_len]
             folderindex = idx // 1000
             imageindex=idx

          if(self.testdataset):
             folderindex=folderindex+90
             imageindex=imageindex+90000
          imagename=self.image_dir+str(folderindex)+"/"+str(imageindex)+'.png'
      #    image=cv2.imread(imagename)
          image = io.imread(imagename)
          image[:3,:3,:]
          image2=image==[255,255,255]
          image=np.all(image2,axis=2)
          image=1-image.astype(np.float32)
        
        #  image = self.rgb2gray(image)
        #  image = np.where(image == 3*255, 0, 1)
          
          #make grayscale image so instead 1000x1000X3 now 1000x1000x1

          #edgesdf = self.compounds.loc[self.compounds['molid']==idx]
         
          edgesdf = self.compounds.loc[idx]

          atoms1df=edgesdf[['atom1','charge1','atom1coord1','atom1coord2']]
          atoms1df.columns=atoms1df.columns.str.replace('atom1','atom')
          atoms1df.columns=atoms1df.columns.str.replace('charge1','charge')

          atoms2df=edgesdf[['atom2','charge2','atom2coord1','atom2coord2']]
          atoms2df.columns=atoms2df.columns.str.replace('atom2','atom')
          atoms2df.columns=atoms2df.columns.str.replace('charge2','charge')

          atomframes=[atoms1df,atoms2df]
          result=pd.concat(atomframes)
          atoms=result.drop_duplicates().values
          x = atoms[:,2:].astype(np.float32)
          some = pdist(x)
          if some.min()<15:
             print(str(imagename))
        
#          import ipdb; ipdb.set_trace()
          target_atom=np.zeros(image.shape)
          target_bond=np.zeros(image.shape)
          target_charge=np.zeros(image.shape)
         # target=np.full((1000, 1000), 255)
          stereo_lst = []
          edgesdf = edgesdf[edgesdf.bondtype != "nobond"]
          #loop over all edges and draw the lines and label them correct in target
          if self.with_edges:
             for index, row in edgesdf.iterrows():
                 #print row['c1'], row['c2']
                 if float(row['bondtype']) >= 4:
                    a = float(row['bondtype'])
                    b = Decimal(a).quantize(0, ROUND_HALF_UP)
                    #if .5 so direction triangle is |->
                    if a<b:
                       basepoint_x=int(row['atom1coord2'])
                       basepoint_y=int(row['atom1coord1'])
                       endpoint_x=int(row['atom2coord2'])
                       endpoint_y=int(row['atom2coord1'])
                       diff_add=-0.5
                    #if .0 so direction triangle is <-|
                    else:
                       basepoint_x=int(row['atom2coord2'])
                       basepoint_y=int(row['atom2coord1'])
                       endpoint_x=int(row['atom1coord2'])
                       endpoint_y=int(row['atom1coord1'])
                       diff_add=0

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
                    target_bond[rr, cc]=int(dict_bond[str(float(row['bondtype'])+diff_add)])
 
                    rr, cc = draw.line(endpoint_x, endpoint_y, centerx_endpoint, centery_endpoint)
                    target_bond[rr, cc]=int(dict_bond[str(float(row['bondtype'])+diff_add+0.5)]) 
                 else:
                    rr, cc = draw.line(int(row['atom1coord2']), int(row['atom1coord1']), int(row['atom2coord2']), int(row['atom2coord1']))
                    target_bond[rr, cc]=int(dict_bond[str(row['bondtype'])])
          #       print(rr)
#                 target[rr, cc]=int(dict[str(row['bondtype'])])
                 #print(row['bondtype'])

          numrows=np.size(atoms,0)
          i=0

          while (i<numrows):
                target_atom[int(atoms[i,3]),int(atoms[i,2])]=int(dict_atom.setdefault(atoms[i,0], '14'))
                target_charge[int(atoms[i,3]),int(atoms[i,2])]=int(dict_charge.setdefault(atoms[i,1], '4'))
                #print("coord1 "+str(int(atoms[i,1]))+"coord2 "+str(int(atoms[i,2]))+"value "+str(int(dict[atoms[i,0]]))+"\n")
                i=i+1
          
          target_atom = ndimage.maximum_filter(target_atom,size=self.filter_size)
          target_bond = ndimage.maximum_filter(target_bond,size=self.filter_size)
          target_charge = ndimage.maximum_filter(target_charge,size=self.filter_size)
         
         # target = ndimage.minimum_filter(target,size=10)
         # np.place(target, target>22, 0)
          imagetensor = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
          #imagetensor = torch.from_numpy(image.astype(np.float32))
          targettensor_atom = torch.from_numpy(target_atom.astype(np.int64))
          targettensor_bond = torch.from_numpy(target_bond.astype(np.int64))
          targettensor_charge = torch.from_numpy(target_charge.astype(np.int64))
          sample = {'image': imagetensor, 'target_atom': targettensor_atom, 'target_bond': targettensor_bond, 'target_charge': targettensor_charge} 
          if self.debug:
             self.create_atom_list(atoms)
             self.create_bond_list(edgesdf)
          #   sample = {'image': imagetensor, 'target_atom': targettensor_atom, 'target_bond': targettensor_bond, 'target_charge': targettensor_charge, 'atom_list': atom_list, 'bond_list': bond_list}
          #sample = {'image': imagetensor, 'target_atom': targettensor_atom, 'target_bond': targettensor_bond} 
          #sample = {imagetensor, targettensor} 
          
 
          return sample
 
      def rgb2gray(self,rgb):
          return np.dot(rgb[...,:3], [1, 1, 1])
