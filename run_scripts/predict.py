import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from skimage import io, transform, color
from scipy import ndimage
import sys, getopt
from rdkit import Chem
from rdkit.Chem import Draw
import skimage.draw as draw
from itertools import *
from math import sqrt
from scipy.spatial.distance import pdist
import pandas as pd
#from utils_graph import *
from ChemGrapher import *
#from ChemGrapher.model import *
import ntpath
import argparse

parser = argparse.ArgumentParser(description="Make graph prediction given input image - check if correct given labelled dataset")
list_of_choices = ["smiles", "planar"]
parser.add_argument("--type", help="Predict SMILES (smiles) or planar (planar) embedding?", type=str, default="smiles", choices=list_of_choices)
parser.add_argument("--inputfile", help="Input image with compound to predict", type=str, default="1K_images/0/0.png")
parser.add_argument("--segnetwork", help="Segmentation Network to use", type=str, required=True)
parser.add_argument("--clas_bond_network", help="Bond Classification Network to use", type=str, required=True)
parser.add_argument("--clas_atom_network", help="Atom Classification Network to use", type=str, required=True)
parser.add_argument("--clas_charge_network", help="Charge Classification Network to use", type=str, required=True)
parser.add_argument("--original_smiles", help="If available specify the original smiles, if prediction is correct labels will be created", type=str, default=None)
parser.add_argument("--trycorrect", help="Set this to one if you want to actively search for correcting small mistakes", type=int, default=0)
args = parser.parse_args()

inputfile = args.inputfile
segnetwork = args.segnetwork
print(f"Using segmentation model {segnetwork}")
clas_bond_network = args.clas_bond_network
print(f"Using Bond Classification model {clas_bond_network}")
clas_atom_network = args.clas_atom_network
print(f"Using Atom Classification model {clas_atom_network}")
clas_charge_network = args.clas_charge_network
print(f"Using Charge Classification model {clas_charge_network}")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")                                                                                                                                                                     
#print(device)

mycnn_seg = MyCNN5(numclasses=28)
mycnn_seg.load_state_dict(torch.load(segnetwork, map_location=device))

mycnn_clasbond = PredictBond3(numclasses=8, inputdim=11)
mycnn_clasbond.load_state_dict(torch.load(clas_bond_network, map_location=device))

mycnn_clasatom = PredictPos7(numclasses=15, inputdim=17)
mycnn_clasatom.load_state_dict(torch.load(clas_atom_network, map_location=device))

mycnn_clascharge = PredictPos7(numclasses=5, inputdim=7)
mycnn_clascharge.load_state_dict(torch.load(clas_charge_network, map_location=device))

dict_atom = {0: 'empty', 1: 'C', 2: 'H', 3: 'N', 4: 'O', 5: 'S', 6: 'F', 7: 'Cl', 8: 'Br', 9: 'I', 10: 'Se', 11: 'P', 12: 'B', 13: 'Si', 14:'Si'}
dict2 = {'0.0': 0, '1.0': 1, '2.0': 2, '3.0': 3, '4.0': 4, '4.5': 5, '5.0': 6, '5.5': 7}
dict_charge = {0: 0, 1: -1, 2: 0, 3: 1, 4: 2}

with torch.no_grad():
   image = io.imread(inputfile)
   image = image[:,:,:3]
   image[:3,:3,:]
   image2=image==[255,255,255]
   image=np.all(image2,axis=2)
   image=1-image.astype(np.float32)
   imagetensor = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
   output_seg = mycnn_seg(imagetensor.unsqueeze(0))
   output1 = output_seg[:,:15]
   output2 = output_seg[:,15:23]
   output3 = output_seg[:,23:]
   bond_max_last = torch.argmax(output2.cpu(), dim=1)
   center_list = generateCenterList(output1.cpu(), output3.cpu())

##### PROCESS ATOM CANIDATES
   atom_list = []
   for centers in center_list:
          x=int(centers[0])
          y=int(centers[1])
          input_tensor2 = LastLayerDataset_clas_atom.get_input_tensor(output1.cpu().squeeze(0).numpy(),image,x,y)
          input_tensor_forcharge = LastLayerDataset_clas_atom.get_input_tensor(output3.cpu().squeeze(0).numpy(),image,x,y)

          output_clasatom = mycnn_clasatom(input_tensor2.unsqueeze(0))
          output_clascharge = mycnn_clascharge(input_tensor_forcharge.unsqueeze(0))
          maxoutput=torch.argmax(output_clasatom, dim=1)
          maxoutput_charge=torch.argmax(output_clascharge, dim=1)
          if maxoutput > 0:
                a=[x,y,int(maxoutput),int(maxoutput_charge)]
                atom_list.append(a)
   
   atom_list_np=np.array(atom_list)
##### END PROCESS ATOM CANDIATES

#### GENERATE BOND CANIDATES
   some = pdist(atom_list_np[:,:2])
   list_comb=list(combinations(range(len(atom_list_np)),2))
#write list of atoms with atom type
   
#this has to be checked with true list of atoms (double ways to check falses etc)
   cand_list = [ list_comb[i] for i in np.nonzero(some<100)[0]]
   coord_cand_list = [ [[atom_list_np[i[0]]],[atom_list_np[i[1]]]] for i in cand_list] 
#### END GENERATE BOND CANIDATES
   
#### BUILD PART OF RESULTING MOLECULE 
   mol = Chem.RWMol()
   m = Chem.Mol()
   em = Chem.EditableMol(m)
   hash_list_atoms={}
   hash_list_atoms_obj={}
   hash_list_atoms_translate={}
   for atom in atom_list:
       a = Chem.Atom(dict_atom[atom[2]]) 
       a.SetProp("X", str(atom[0])) 
       a.SetProp("Y", str(atom[1]))
       hash_list_atoms_obj[str(atom[0])+"_"+str(atom[1])]=a 
       idx = em.AddAtom(a) 
       hash_list_atoms[str(atom[0])+"_"+str(atom[1])]=idx 
#### END BUILD PART OF RESULTING MOLECULE

#### PROCESS BOND CANDIDATES
   bond_list = []
   out_idx = 0
   #for cand_pair in cand_list:
   for cand_pair in coord_cand_list:
        other_atoms=np.zeros((1000, 1000))
        cand_line=np.zeros((1000,1000))
        numrows=np.size(atom_list_np,0)
        i=0
        while (i<numrows):
            other_atoms[int(atom_list_np[i,0]),int(atom_list_np[i,1])]=1
            i=i+1
        
        x1=int(cand_pair[0][0][0])
        y1=int(cand_pair[0][0][1])
        x2=int(cand_pair[1][0][0])
        y2=int(cand_pair[1][0][1]) 

        other_atoms[x1,y1]=0
        other_atoms[x2,y2]=0
        other_atoms = ndimage.maximum_filter(other_atoms,size=10)
        rr, cc = draw.line(x1, y1, x2, y2)
        cand_line[rr, cc]=1
        m = np.where(other_atoms==1, cand_line, 0)
      #  masked_x = np.ma.compressed(m)
      #  mask = (other_atoms == 1 and cand_line == 1)
        if not 1 in m:
          output2_np=output2.cpu().squeeze(0).numpy()
          input_tensor3=BondLayerDataset.get_input_tensor(output2_np,image,x1,y1,x2,y2)
          output_clasbond = mycnn_clasbond(input_tensor3.unsqueeze(0))

          maxoutput=torch.argmax(output_clasbond, dim=1)
          if maxoutput > 0:
             a=[cand_pair[0],cand_pair[1],int(maxoutput)]
             bond_list.append(a)
#### END PROCESS BOND CANIDATES
   #
#### BUILD RESULTING MOLECULE AND FILTER BONDS IF NECESSARY
   results_bond = []
   for bond in bond_list:
       idx1 = hash_list_atoms[str(bond[0][0][0])+"_"+str(bond[0][0][1])]
       idx2 = hash_list_atoms[str(bond[1][0][0])+"_"+str(bond[1][0][1])]
       a1 = hash_list_atoms_obj[str(bond[0][0][0])+"_"+str(bond[0][0][1])]
       a2 = hash_list_atoms_obj[str(bond[1][0][0])+"_"+str(bond[1][0][1])]
  #     print(bond)
  #     print(str(bond[0][0][0])+"_"+str(bond[0][0][1]))
       if bond[2] == 1 or bond[2] >= 4:
          bondIdx = em.AddBond(idx1,idx2, Chem.BondType.SINGLE)
       if bond[2] == 2:
          bondIdx = em.AddBond(idx1,idx2, Chem.BondType.DOUBLE)
       if bond[2] == 3:
          bondIdx = em.AddBond(idx1,idx2, Chem.BondType.TRIPLE)
       
       m = em.GetMol()
       if Chem.SanitizeMol(m, catchErrors = True) != 0:
           em.RemoveBond(idx1, idx2)
       else:
           results_bond.append(bond)
   bond_list = results_bond
   m = em.GetMol()
#### END BUILD RESULTING MOLECULE AND FILTER BONDS IF NECESSARY

#   print(f"(PREPROCESSED) SMILES FROM {inputfile}: {Chem.MolToSmiles(m)}")
   atom_list = filter_nobonds(atom_list, bond_list) 
   file_path2 = "unfiltered_molfiletest"
   chiral = 0
   if check_chiral(bond_list):
        chiral = 1
   save_mol_to_file(atom_list, bond_list, file_path2, chiral)
   predicted_mol = Chem.MolFromMolFile("unfiltered_molfiletest")
   if args.original_smiles is not None:
      #original_smiles = "CC(C)(C)C1C=CC(=CC=1)OCC[C@H](CN)N1CCCCC1"
      original_smiles = args.original_smiles
      filtered_atom_list, filtered_bond_list = filter_graph(atom_list, bond_list, original_smiles, m)
      label_predict_smiles(original_smiles, filtered_atom_list, filtered_bond_list, inputfile)
   
#   print(f"(FILTERED) SMILES FROM {inputfile}: {Chem.MolToSmiles(predicted_mol)}")
   if args.type == "smiles":
      print(f"{Chem.MolToSmiles(predicted_mol)}")
   else:
      print(atom_list)
      print(bond_list)


