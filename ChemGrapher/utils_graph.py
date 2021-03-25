import torch
import numpy as np
import os
from skimage import io, transform, color
from scipy import ndimage
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import rdFMCS
from math import sqrt
import copy
import ntpath
import requests
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.lines import Line2D

#colormap = {'0': (1,1,1), '1': (0,0,0), '2': (1,0.65,0), '3': (0.18,0.31,0.97), '4': (1,0.05,0.05), '5': (1,1,0.19), '6': (0.56,0.88,0.31), '7': (0.12,0.94,0.12), '8': (0.65,0.16,0.16), '9': (0.58,0,0.58), '10': (0.08,0,0.58), '11': (0.15,0.15,0.15), '12': (0.07,1,0.19), '13': (0.18,0.31,0.97), '14': (0.80,0.25,0.25), '15': (1,1,1), '16': (0.56,0.88,0.31), '17': (0.12,0.94,0.12), '18': (0.65,0.16,0.16), '19': (0.58,0,0.58), '20': (0.08,0,0.58), '21': (0.08,0,0.58), '22': (0.08,0,0.58)}
colormap = {'0': (1,1,1), '1': (0,0,0), '2': (1,0.65,0), '3': (0.18,0.31,0.97), '4': (1,0.05,0.05), '5': (1,1,0.19), '6': (0.56,0.88,0.31), '7': (0.12,0.94,0.12), '8': (0.65,0.16,0.16), '9': (0.58,0,0.58), '10': (0.08,0,0.58), '11': (0.15,0.15,0.15), '12': (0.07,1,0.19), '13': (0.18,0.31,0.97), '14': (0,0,0), '15': (1,1,1), '16': (0.56,0.88,0.31), '17': (0.12,0.94,0.12), '18': (0.65,0.16,0.16), '19': (0.58,0,0.58), '20': (0.08,0,0.58), '21': (0.08,0,0.58), '22': (0.08,0,0.58)}

dict_atom = {0: 'empty', 1: 'C', 2: 'H', 3: 'N', 4: 'O', 5: 'S', 6: 'F', 7: 'Cl', 8: 'Br', 9: 'I', 10: 'Se', 11: 'P', 12: 'B', 13: 'Si', 14:'*'}
dict_atom_rev = {'C': 1, 'H': 2, 'N': 3, 'O': 4, 'S': 5, 'F': 6, 'Cl': 7, 'Br': 8, 'I': 9, 'Se': 10, 'P': 11, 'B': 12, 'Si': 13, '*':14}
#dict2 = {'0.0': 0, '1.0': 1, '2.0': 2, '3.0': 3, '4.0': 4, '4.5': 5, '5.0': 6, '5.5': 7}
dict_bond = {0:'0.0', 1:'1.0', 2: '2.0', 3: '3.0', 4: '4.0', 5: '4.5', 6: '5.0', 7: '5.5'}
dict_charge = {0: 0, 1: -1, 2: 0, 3: 1, 4: 2}

def compute_graph_metrics(labels, preds):  

     preds_max = torch.argmax(preds, dim=1)
     TP = torch.stack([(labels==i) & (preds_max==i) for i in range(preds.shape[1])], dim=1)
     Pos_Pred = torch.stack([(preds_max==i) for i in range(preds.shape[1])], dim=1 )
     Pos = torch.stack([(labels==i) for i in range(preds.shape[1])], dim=1 )

     intersection =torch.stack([(labels==i) & (preds_max==i) for i in range(preds.shape[1])], dim=1)
     union = torch.stack([(labels==i) | (preds_max==i) for i in range(preds.shape[1])], dim=1 )
 #    intersection =torch.stack([(labels==i) & (atom_max==i) for i in range(output.shape[1])], dim=1)  
 #    union = torch.stack([(labels==i) | (atom_max==i) for i in range(output.shape[1])], dim=1 )
#     intersection_sum = intersection.sum(0).sum(0).sum(0)
     TP_sum = TP.sum(dim=(2,3))
    # TP_sum_list.append(TP_sum.cpu().numpy())

     Pos_pred_sum = Pos_Pred.sum(dim=(2,3))
    # Pos_Pred_atoms_list.append(Pos_pred_atoms_sum.cpu().numpy())

     Pos_sum = Pos.sum(dim=(2,3))
    # Pos_atoms_list.append(Pos_atoms_sum.cpu().numpy())

     intersection_sum = intersection.sum(dim=(2,3))
    # intersection_atoms_sum_list.append(intersection_atoms_sum.cpu().numpy())
     union_sum = union.sum(dim=(2,3))
    # union_atoms_sum_list.append(union_atoms_sum.cpu().numpy())
    # import ipdb; ipdb.set_trace()
     iou_per_image = (intersection_sum.type(torch.cuda.FloatTensor)/union_sum.type(torch.cuda.FloatTensor))
    # iou_atoms = iou_atoms_per_image.cpu().numpy()
    # iou_atoms_list.append(iou_atoms)
    # iou_atoms_sum+=intersection_atoms_sum.cpu().numpy().sum()/union_atoms_sum.cpu().numpy().sum()
     metrics = (intersection_sum, union_sum, iou_per_image, TP_sum, Pos_pred_sum, Pos_sum)
     return metrics

def label_atoms_bonds(atom_list, bond_list, file_path, index):
    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
        file = open(file_path,"a")
    else:
        file = open(file_path,"w")
        file.write("index,molid,bondtype,atom1,charge1,atom2,charge2,atom1coord1,atom1coord2,atom2coord1,atom2coord2\n")
    linenumber=0
    hash_list_atoms={}
    for i,atom in enumerate(atom_list):
        atomc1=atom[1]
        atomc2=atom[0]
        hash_list_atoms[str(atomc1)+"_"+str(atomc2)] = i
        if atom[2] != 1:
           file.write(f"{index}{linenumber},{index},nobond,{dict_atom[atom[2]]},{dict_charge[atom[3]]},{dict_atom[atom[2]]},{dict_charge[atom[3]]},{atom[1]},{atom[0]},{atom[1]},{atom[0]}\n")
           linenumber += 1
    #file.write(str(bond_list))
    for j,bond in enumerate(bond_list):
        #file.write(f"{str(bond)}\n")
        atom1 = atom_list[hash_list_atoms[str(bond[0][0][1])+"_"+str(bond[0][0][0])]][2]
        atom2 = atom_list[hash_list_atoms[str(bond[1][0][1])+"_"+str(bond[1][0][0])]][2]
        #file.write(f"{index}{linenumber+j+1},{index},{dict_bond[bond[2]]},{dict_atom[bond[0][0][2]]},{dict_charge[bond[0][0][3]]},{dict_atom[bond[1][0][2]]},{dict_charge[bond[1][0][3]]},{bond[0][0][1]},{bond[0][0][0]},{bond[1][0][1]},{bond[1][0][0]}\n")
        file.write(f"{index}{linenumber+j+1},{index},{dict_bond[bond[2]]},{dict_atom[atom1]},{dict_charge[bond[0][0][3]]},{dict_atom[atom2]},{dict_charge[bond[1][0][3]]},{bond[0][0][1]},{bond[0][0][0]},{bond[1][0][1]},{bond[1][0][0]}\n")
        #,{dict_charge[bond[0][3]]}")
        #,{dict_atom[bond[1][2]]},{dict_charge[bond[1][3]]},{bond[0][0]},{bond[0][1]},{bond[1][0]},{bond[1][1]}")
    return 0

def check_bond(bond_list, number):
    atom1_str = str(bond_list[number][0][0][1])+"_"+str(bond_list[number][0][0][0])
    atom2_str = str(bond_list[number][1][0][1])+"_"+str(bond_list[number][1][0][0])
    atom1_count = 0
    atom2_count = 0
    for j,bond in enumerate(bond_list):
        tmp_atom_list = [str(bond[0][0][1])+"_"+str(bond[0][0][0]), str(bond[1][0][1])+"_"+str(bond[1][0][0])]
        if atom1_str in tmp_atom_list:
           atom1_count += 1
        
        if atom2_str in tmp_atom_list:
           atom2_count += 1

    if atom1_count == 1:
       allowed = 1
       direction = 0
    elif atom2_count == 1:
       allowed = 1
       direction = 1
    else:
       allowed = 0
       direction = 0

    return allowed,direction

def save_tensor_to_image(path, input1, input2=None, same_dim=False):
   values = np.unique(input1.cpu().numpy())
   colors_map=[]
   for i in np.nditer(values):
       colors_map.append(colormap[str(i)])
   
   if input2 is not None:
       if same_dim:
          output=color.label2rgb(input1.cpu().numpy()[0,:,:],input2.cpu().numpy()[0,:,:],colors=colors_map)
       else:
          output=color.label2rgb(input1.cpu().numpy()[0,:,:],input2.cpu().numpy()[0,0,:,:],colors=colors_map)
   else:
       output=color.label2rgb(input1.cpu().numpy()[0,:,:],colors=colors_map)
   io.imsave(path,output)

def read_mol_from_file(molfile_path, imagefile_path):
    molfile = open(molfile_path, "r")
    imagefile = io.imread(imagefile_path, plugin='pil')
    xdim = imagefile.shape[0]
    ydim = imagefile.shape[1]
    print(f"shape is {imagefile.shape}")
    mollines = molfile.readlines()
    atom_number = 1
    left_border = 0
    atom_hash = {}
    atom_list = []
    bond_list = []
    for i,line in enumerate(mollines):
        if i == 3: #header line
           split_line = line.split()
           numatoms = int(split_line[0])
           numbonds = int(split_line[1])
        if i > 3:
            if i <= 3+numatoms:#atomline
                split_line = line.split()
                atom_x = float(split_line[0])
                atom_y = float(split_line[1])
                atom_type = split_line[3]
                if atom_x < left_border:
                    left_border = atom_x
                atom_hash[atom_number] = [atom_x, atom_y, atom_type]
                atom_number = atom_number + 1
            if (i > 3+numatoms) and (i <= 3+numatoms+numbonds):#bondline
                split_line = line.split()
                atom1 = int(split_line[0])
                atom2 = int(split_line[1])
                atom1coords =  atom_hash[atom1]
                atom2coords =  atom_hash[atom2]
                offset = -(xdim/2)/left_border
                atom1coord1 = xdim/2 + atom1coords[0]*offset + 1
                atom1coord2 = ydim/2 - (atom1coords[1]*offset - 1)
                atom2coord1 = xdim/2 + atom2coords[0]*offset +1
                atom2coord2 = ydim/2 - (atom2coords[1]*offset -1)
                bond = [[[atom1coord2,atom1coord1,dict_atom_rev[atom1coords[2]],2]],[[atom2coord2,atom2coord1,dict_atom_rev[atom2coords[2]],2]],int(split_line[2])]
                print(f"adding bond between {atom1}: {atom1coord1},{atom1coord2} and {atom2}:{atom2coord1},{atom2coord2}, offset = {offset}")
                bond_list.append(bond)
    for key in atom_hash:
        atomcoords = atom_hash[key]
        offset = -(xdim/2)/left_border
        atomcoord1 = xdim/2 + atomcoords[0]*offset +1
        atomcoord2 = ydim/2 - (atomcoords[1]*offset-1)
        print(f"adding atom {key}: {atomcoord1},{atomcoord2} , offset = {offset}")
        atom = [atomcoord2, atomcoord1, dict_atom_rev[atomcoords[2]], 2]
        atom_list.append(atom)

    print(atom_list)
    print(bond_list)
    
    return atom_list, bond_list



def save_mol_to_file(atom_list, bond_list, file_path, chiral=0):
    file = open(file_path,"w")
    file.write("\n")
    file.write(" MOLFILE FROM CHEMGRAPHER\n")
    file.write("\n")
    num_atoms = len(atom_list)
    num_bonds = len(bond_list)
    hash_list_atoms={}
    file.write(f"{num_atoms:3.0f}{num_bonds:3.0f}  0  0  {chiral}  0            999 V2000\n")
    
    for i,atom in enumerate(atom_list):
       # atom1 = (atom[1]-500)/250
        atom1 = (atom[1]-500)
        atom2 = (atom[0]-500)*(-1)
       # atom2 = (atom[0]-500)/250
        zero = 0.0

        file.write(f"{atom1:10.4f}{atom2:10.4f}{zero:10.4f}  {dict_atom[atom[2]]}  0  0\n")
        hash_list_atoms[str(atom1)+"_"+str(atom2)] = i+1

    for j,bond in enumerate(bond_list):
       # atom1_1 = (bond[0][0][1] - 500)/250
       # atom1_2 = (bond[0][0][0] - 500)/250
       # atom2_1 = (bond[1][0][1] - 500)/250
       # atom2_2 = (bond[1][0][0] - 500)/250
        atom1_1 = (bond[0][0][1] - 500)
        atom1_2 = (bond[0][0][0] - 500)*(-1)
        atom2_1 = (bond[1][0][1] - 500)
        atom2_2 = (bond[1][0][0] - 500)*(-1)

        if bond[2] == 5 or bond[2] == 7:
            atom_num1 = hash_list_atoms[str(atom2_1)+"_"+str(atom2_2)]
            atom_num2 = hash_list_atoms[str(atom1_1)+"_"+str(atom1_2)]
        else:
            atom_num1 = hash_list_atoms[str(atom1_1)+"_"+str(atom1_2)]
            atom_num2 = hash_list_atoms[str(atom2_1)+"_"+str(atom2_2)]

        if bond[2] >= 4:
            bond_num = 1
            if chiral == 1:
  #             if bond[2] == 4 or bond[2] == 7:
               if bond[2] == 4 or bond[2] == 5:
                  bond_dir = 1
               else:
                  bond_dir = 6
            else:
                bond_dir = 0
        else:
            bond_num = bond[2]
            bond_dir = 0

        file.write(f"{atom_num1:3.0f}{atom_num2:3.0f}{bond_num:3.0f}{bond_dir:3.0f}\n")

    for i,atom in enumerate(atom_list):
        if atom[3] != 2:
            file.write(f"M  CHG  1{i+1:3.0f}{dict_charge[atom[3]]:3.0f}\n")

    file.write(f"M  END")

def filter_graph(atom_list, bond_list, original_smiles, pred_mol):
    filtered_atom_list = []
    filtered_bond_list = []

    original_mol = Chem.MolFromSmiles(original_smiles)
    if original_mol is not None:
    #todo filter
       chiral = 0
       if check_chiral(bond_list):
           chiral = 1
       hash_atoms = {}
       filtered_hash_atoms = {}
       for atom in atom_list:
           hash_atoms[str(atom[0])+"_"+str(atom[1])] = atom
       original_mol = Chem.MolFromSmiles(original_smiles)
       Chem.SanitizeMol(pred_mol)
       mols = [original_mol, pred_mol]
       res=rdFMCS.FindMCS(mols)
       patt = Chem.MolFromSmarts(res.smartsString)
       sub_struct = pred_mol.GetSubstructMatch(patt)
       for idx in sub_struct:
           atom_check = pred_mol.GetAtomWithIdx(idx)
           filtered_atom_list.append(hash_atoms[atom_check.GetProp("X")+"_"+atom_check.GetProp("Y")])
           filtered_hash_atoms[atom_check.GetProp("X")+"_"+atom_check.GetProp("Y")] = 1
       for bond in bond_list:
           first_atom = str(bond[0][0][0])+"_"+str(bond[0][0][1])
           second_atom = str(bond[1][0][0])+"_"+str(bond[1][0][1])

           if first_atom in filtered_hash_atoms:
               if second_atom in filtered_hash_atoms:
                   filtered_bond_list.append(bond)
    else:
        filtered_atom_list = atom_list
        filtered_bond_list = bond_list

    return filtered_atom_list, filtered_bond_list

def filter_nobonds(atom_list, bond_list):
    filtered_atoms = []
    no_bonds = get_nobond_atoms(atom_list, bond_list)
    avg_bond_len = get_average_bond_length(bond_list)
    for atom in atom_list:
        #if atom in no_bonds and atom[2] == 2:
        if atom in no_bonds:
           min_dist = get_minimal_distance(atom_list, atom)
           if min_dist > 0.6 * avg_bond_len:
              filtered_atoms.append(atom)
        else:
           filtered_atoms.append(atom)
    return filtered_atoms

def get_minimal_distance(atom_list, cand_atom):
    min_dist = 1000
    for atom in atom_list:
        dist = sqrt(pow((atom[0]-cand_atom[0]),2) + pow((atom[1]-cand_atom[1]),2))
        if dist < min_dist:
           min_dist = dist
    return min_dist

def get_nobond_atoms(atom_list, bond_list):
    nobond_atoms = []
    hash_bond_atoms = {}
    for bond in bond_list:
        hash_bond_atoms[str(bond[0][0][0])+"_"+str(bond[0][0][1])] = 1
        hash_bond_atoms[str(bond[1][0][0])+"_"+str(bond[1][0][1])] = 1
    for atom in atom_list:
        X_Y = str(atom[0])+"_"+str(atom[1])
        if X_Y not in hash_bond_atoms:
           nobond_atoms.append(atom)
    return nobond_atoms

def get_average_bond_length(bond_list):
    distances = []
    for bond in bond_list:
        atom1_x = int(bond[0][0][0])
        atom1_y = int(bond[0][0][1])
        atom2_x = int(bond[1][0][0])
        atom2_y = int(bond[1][0][1])
        #import ipdb; ipdb.set_trace()
        dist = sqrt(pow((atom1_x-atom2_x),2) + pow((atom1_y-atom2_y),2))
        distances.append(dist)
    if len(distances)>0:
       return sum(distances)/len(distances)
    else:
       return 0

def alter_graph(toalter_list, notaltered_list, original_smiles, filesuffix="", type_alter=0, replacenitrogen=0):
    ignore_list = [0]
    if type_alter == 0:
        bond_list = notaltered_list
        atom_list = toalter_list
    else:
        bond_list = toalter_list
        atom_list = notaltered_list
    if type_alter == 0:
       dictionary = dict_atom
    else:
       dictionary = dict_bond
    num_alters = len(toalter_list)
    original_mol = Chem.MolFromSmiles(original_smiles)
    if original_mol is not None:
       smiles_original = Chem.MolToSmiles(original_mol)
    else:
       smiles_original = original_smiles
    for i in range(num_alters):
        altered_list = copy.deepcopy(toalter_list)
        if type_alter == 1:
           allowed,direction = check_bond(toalter_list,i)
          # import ipdb; ipdb.set_trace()
          #todo setup ignore_list correctly depending on allowed and direction
           if allowed == 0:
               ignore_list = [0,4,5,6,7]
           else:
               if direction == 0:
                   ignore_list = [0,4,6] #tocheck
               else:
                   ignore_list = [0,5,7] #tocheck
        for type_key in dictionary:
            if type_key not in ignore_list:
                altered_list[i][2] = type_key
                if type_alter == 0:
                    atom_list = altered_list
                else:
                    bond_list = altered_list
                file_path2 = f"molfiletest{filesuffix}"
                chiral = 0
                if check_chiral(bond_list):
                   chiral = 1
                save_mol_to_file(atom_list, bond_list, file_path2, chiral)
                predicted_mol = Chem.MolFromMolFile(f"molfiletest{filesuffix}")
                if predicted_mol is not None:
                   if replacenitrogen == 1:
                     nitrogen = Chem.MolFromSmiles("[N+](=O)[O-]")
                     replaced = Chem.rdmolops.ReplaceSubstructs(predicted_mol,Chem.MolFromSmiles("*"),nitrogen,replaceAll=True)
                     predicted_mol = replaced[0]
                   #  smiles_pred_mol = Chem.MolToSmiles(predicted_mol)
                   try:
                     smiles_pred_mol = Chem.MolToSmiles(predicted_mol)
                   except RuntimeError:
                     smiles_pred_mol = "predicted_mol"
                else:
                   smiles_pred_mol = "predicted_mol"
                smiles_noslashes = smiles_pred_mol
                smiles_noslashes = smiles_noslashes.replace("\\","")
                smiles_noslashes = smiles_noslashes.replace("/","")
                if smiles_pred_mol == smiles_original or smiles_noslashes == smiles_original:
                   return altered_list
    return None
                   
''' 
    for i in range(num_alters-1):
        for j in range(i+1,num_alters):
            altered_list = copy.deepcopy(toalter_list)
            for type_key in dictionary:
                if type_key not in ignore_list:
                   altered_list[i][2] = type_key
                   altered_list[j][2] = type_key
                   if type_alter == 0:
                      atom_list = altered_list
                   else:
                      bond_list = altered_list
                   file_path2 = f"molfiletest{filesuffix}"
                   chiral = 0
                   if check_chiral(bond_list):
                      chiral = 1
                   save_mol_to_file(atom_list, bond_list, file_path2, chiral)
                   predicted_mol = Chem.MolFromMolFile(f"molfiletest{filesuffix}")
                   if predicted_mol is not None:
                      try:
                        smiles_pred_mol = Chem.MolToSmiles(predicted_mol)
                      except RuntimeError:
                        smiles_pred_mol = "predicted_mol"
                   else:
                      smiles_pred_mol = "predicted_mol"
                   smiles_noslashes = smiles_pred_mol
                   smiles_noslashes = smiles_noslashes.replace("\\","")
                   smiles_noslashes = smiles_noslashes.replace("/","")
                   if smiles_pred_mol == smiles_original or smiles_noslashes == smiles_original:
                      return altered_list
'''
   # return None




def alter_atom_graph(atom_list, bond_list, original_smiles, filesuffix="", replacenitrogen=0):
    ignore_list = [0]
    num_atoms = len(atom_list)
    original_mol = Chem.MolFromSmiles(original_smiles)
    if original_mol is not None:
       smiles_original = Chem.MolToSmiles(original_mol)
    else:
       smiles_original = original_smiles
    for i in range(num_atoms):
        altered_list = copy.deepcopy(atom_list)
        for atom_type_key in dict_atom:
            if atom_type_key not in ignore_list:
               altered_list[i][2] = atom_type_key
            #   print(altered_list)
             #  print(atom_list)
               #check if correct now, if correct STOP and return altered_list
               file_path2 = f"molfiletest{filesuffix}"
               chiral = 0
               if check_chiral(bond_list):
                  chiral = 1
               save_mol_to_file(altered_list, bond_list, file_path2, chiral)
               predicted_mol = Chem.MolFromMolFile(f"molfiletest{filesuffix}")
               if predicted_mol is not None:
                  if replacenitrogen == 1:
                     nitrogen = Chem.MolFromSmiles("[N+](=O)[O-]")
                     replaced = Chem.rdmolops.ReplaceSubstructs(predicted_mol,Chem.MolFromSmiles("*"),nitrogen,replaceAll=True)
                     predicted_mol = replaced[0]
                  try:
                    smiles_pred_mol = Chem.MolToSmiles(predicted_mol)
                  except RuntimeError:
                    smiles_pred_mol = "predicted_mol"
               else:
                  smiles_pred_mol = "predicted_mol"
               smiles_noslashes = smiles_pred_mol
               smiles_noslashes = smiles_noslashes.replace("\\","")
               smiles_noslashes = smiles_noslashes.replace("/","")
               if smiles_pred_mol == smiles_original or smiles_noslashes == smiles_original:
                  return altered_list
               #file_path="labelled_graph_output"
               #label_atoms_bonds(filtered_atom_list, filtered_bond_list, file_path, molid)
               #print(f"SMILES FROM {molid}: CORRECT")
    for i in range(num_atoms-1):
        for j in range(i+1,num_atoms):
            altered_list = copy.deepcopy(atom_list)
            for atom_type_key in dict_atom:
                if atom_type_key not in ignore_list:
                   altered_list[i][2] = atom_type_key
                   altered_list[j][2] = atom_type_key
                   file_path2 = f"molfiletest{filesuffix}"
                   chiral = 0
                   if check_chiral(bond_list):
                      chiral = 1
                   save_mol_to_file(altered_list, bond_list, file_path2, chiral)
                   predicted_mol = Chem.MolFromMolFile(f"molfiletest{filesuffix}")
                   if predicted_mol is not None:
                      if replacenitrogen == 1:
                         nitrogen = Chem.MolFromSmiles("[N+](=O)[O-]")
                         replaced = Chem.rdmolops.ReplaceSubstructs(predicted_mol,Chem.MolFromSmiles("*"),nitrogen,replaceAll=True)
                         predicted_mol = replaced[0]
                      try:
                        smiles_pred_mol = Chem.MolToSmiles(predicted_mol)
                      except RuntimeError:
                        smiles_pred_mol = "predicted_mol"
                   else:
                      smiles_pred_mol = "predicted_mol"
                   smiles_noslashes = smiles_pred_mol
                   smiles_noslashes = smiles_noslashes.replace("\\","")
                   smiles_noslashes = smiles_noslashes.replace("/","")
                   if smiles_pred_mol == smiles_original or smiles_noslashes == smiles_original:
                      return altered_list


    return None

def check_chiral(bond_list):
    check_chiral = False
    #todo check
    for bond in bond_list:
        if bond[2] >= 4:
            check_chiral = True
    return check_chiral

def remove_chirality(bond_list):
    filtered_bond_list = []
    for bond in bond_list:
        tmp_bond = bond
        if bond[2] >= 4:
            tmp_bond[2] = 1
        filtered_bond_list.append(tmp_bond)
    return filtered_bond_list

def generateCenterList(atom_seg, charge_seg):
   atom_max_last = torch.argmax(atom_seg, dim=1)
   charge_max_last = torch.argmax(charge_seg, dim=1)
   np_atom_max = atom_max_last.cpu().numpy()[0,:,:]
   np_charge_max = charge_max_last.cpu().numpy()[0,:,:]
   carbon=np.where(np_charge_max > 0, np_charge_max, 0)
   labeled_array, num_features  = ndimage.label(carbon)
   center_list = ndimage.measurements.center_of_mass(carbon, labeled_array,list(range(1, num_features+1)))

  # place center list in empty image, apply filter, define center of mass again : these are the definite centers.
   center_image = np.zeros((1000, 1000))
   for center in center_list:
       x=int(center[0])
       y=int(center[1])
       center_image[x,y]=1

   center_image = ndimage.maximum_filter(center_image,size=11)
   labeled_array, num_features  = ndimage.label(center_image)
   center_list = ndimage.measurements.center_of_mass(center_image, labeled_array,list(range(1, num_features+1)))
   return center_list

def label_predict_smiles(original_smiles, filtered_atom_list, filtered_bond_list, inputfile):
      file_path2 = "predict_molfiletest"
      chiral = 0
      if check_chiral(filtered_bond_list):
        chiral = 1
      save_mol_to_file(filtered_atom_list, filtered_bond_list, file_path2, chiral)
      file_path3 = "non_chiral_molfile"
      save_mol_to_file(filtered_atom_list, filtered_bond_list, file_path3, 0)
      non_chiral_mol = Chem.MolFromMolFile(file_path3)
      predicted_mol = Chem.MolFromMolFile("predict_molfiletest")
      unfiltered_mol = Chem.MolFromMolFile("unfiltered_molfiletest")
      original_mol = Chem.MolFromSmiles(original_smiles)
      smiles_nonchiral = Chem.MolToSmiles(non_chiral_mol)
      smiles_pred_mol = Chem.MolToSmiles(predicted_mol)
      smiles_original = Chem.MolToSmiles(original_mol)
      smiles_noslashes = smiles_pred_mol
      smiles_noslashes = smiles_noslashes.replace("\\","")
      smiles_noslashes = smiles_noslashes.replace("/","")
      print(f"SMILES FROM UNFILTERED_MOL: {Chem.MolToSmiles(unfiltered_mol)}")
      print(f"SMILES FROM ORIGINALMOL: {Chem.MolToSmiles(original_mol)}")
      print(f"(noslashes) SMILES FROM {inputfile}: {smiles_noslashes}")

      if smiles_pred_mol == smiles_original or smiles_noslashes == smiles_original:
         file_path="labelled_predict_output"
         basename=ntpath.basename(inputfile)
         base_split = basename.split('.')
         base_index = int(base_split[0])
         label_atoms_bonds(filtered_atom_list, filtered_bond_list, file_path, base_index)
         print(f"SMILES FROM {inputfile} : CORRECT")
      else:
          if smiles_nonchiral == smiles_original:
              nonchiral_bond_list = remove_chirality(filtered_bond_list)
              file_path="labelled_predict_output"
              basename=ntpath.basename(inputfile)
              base_split = basename.split('.')
              base_index = int(base_split[0])
              label_atoms_bonds(filtered_atom_list, nonchiral_bond_list, file_path, base_index)
              print(f"SMILES (nonchiral) FROM {inputfile}: CORRECT")
          else:
              if args.trycorrect == 1:
                 altered_atom_list = alter_atom_graph(atom_list, bond_list, original_smiles)
                 if altered_atom_list is not None:
                    file_path2 = "alter_molfiletest"
                    chiral = 0
                    if check_chiral(bond_list):
                       chiral = 1
                    save_mol_to_file(altered_atom_list, bond_list, file_path2, chiral)
                    predicted_mol = Chem.MolFromMolFile("alter_molfiletest")
                    if predicted_mol is not None:
                       smiles_pred_mol = Chem.MolToSmiles(predicted_mol)
                       print(f"WOW THIS WORKS {inputfile}: {smiles_pred_mol}")

#taken from this StackOverflow answer: https://stackoverflow.com/a/39225039

def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()
    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)
    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)
    save_response_content(response, destination)    

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768
    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)

def visualize_planar(graph, graph_edges, filename):
    dict_atom = {0: 'empty', 1: 'C', 2: 'H', 3: 'N', 4: 'O', 5: 'S', 6: 'F', 7: 'Cl', 8: 'Br', 9: 'I', 10: 'Se', 11: 'P', 12: 'B', 13: 'Si', 14:'*'}
    dict_bond = {0: 'none', 1: 'blue', 2: 'orange', 3: 'green', 4: 'blue', 5: 'blue', 6: 'blue', 7: 'blue'} #For sake of simplicity, all stereo bonds will be visualized as single bond
    image = mpimg.imread(filename)
    plt.figure(figsize=(20,20))
    plt.imshow(image)
#plt.plot([g[1] for g in graph], [g[0] for g in graph], 'ro')
    plt.scatter([g[1] for g in graph], [g[0] for g in graph],s=150, color='red', alpha=0.3, label='atom predictions')
    for g in graph:
        t = "({})".format(dict_atom[g[2]])
        plt.text(g[1]+5, g[0]-8, t, color='darkred', fontsize=15) 
    for bond in graph_edges:
    #print(bond[1])
        plt.plot([bond[0][1],bond[1][1]],[bond[0][0],bond[1][0]], linewidth=5, color=dict_bond[int(bond[2])], alpha=0.4)

    custom_lines = [Line2D([0], [0], color=dict_bond[1], alpha=0.4, lw=4),
                Line2D([0], [0], color=dict_bond[2], alpha=0.4, lw=4),
                Line2D([0], [0], color=dict_bond[3], alpha=0.4, lw=4),
               Line2D([0], [0], marker='o', color='w', alpha=0.3,
                          markerfacecolor='r', markersize=12)]

#fig, ax = plt.subplots()
#lines = ax.plot(data)
    plt.legend(custom_lines, ['Single Bond', 'Double Bond', 'Triple Bond', 'Atom Prediction'])
    plt.show()
