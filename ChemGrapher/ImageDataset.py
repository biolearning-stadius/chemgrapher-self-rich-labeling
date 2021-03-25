import skimage.io as io
import numpy as np
import torch
import os

class ImageDataset(torch.utils.data.Dataset):
      
      def __init__(self, image_dir="new_images/", offset=0, half=False):
          self.image_dir = image_dir
          self.offset = int(offset)
          self.half = half
      
      def __len__(self):
          length = len([name for name in os.listdir(self.image_dir) if os.path.isfile(f"{self.image_dir}/{name}")])
        #  print(f"length: {length}")
        #  print(f"imagedor: {self.image_dir}")
        #  import ipdb; ipdb.set_trace()
          if self.half:
             length = int(length/2)
          return length
      
      def __getitem__(self, idx):
          imagename=self.image_dir+"/"+str(idx+self.offset)+'.png'
          image = io.imread(imagename)
          image[:3,:3,:]
          image2=image==[255,255,255]
          image=np.all(image2,axis=2)
          image=1-image.astype(np.float32)
          imagetensor = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
          sample = {'image': imagetensor} 
          return sample
