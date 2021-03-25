from rdkit import Chem
from rdkit.Chem.Draw.MolDrawing import *
from bs4 import BeautifulSoup as BS
import rdkit.Chem.Draw.MolDrawing
import math
import re

#file = open("testfile.txt","a")
class MolDrawing(MolDrawing):
     # file = open("molecules_chiral_triple.txt","a") 
      file = open("datasets/balanced_labelled_data_test.txt","a") 
     # file = open("extra_molecules.txt","a") 
      file.write("index,molid,bondtype,atom1,charge1,atom2,charge2,atom1coord1,atom1coord2,atom2coord1,atom2coord2\n")
      #def __init__(self, canvas=None, drawingOptions=None, filename=None):
      #    super(MolDrawing , self ).__init__(canvas, drawingOptions)
      #    #self["name"] = filename
      #    file = open("testfile.txt","a")
      index=0
      linenumber=1

      def _drawBond(self, bond, atom, nbr, pos, nbrPos, conf, width=None, color=None, color2=None, labelSize1=None, labelSize2=None):
          super(MolDrawing, self)._drawBond(bond, atom, nbr, pos, nbrPos, conf, width, color, color2, labelSize1, labelSize2)
         # file = open("testfile.txt","a") 
 
          #print(bond.GetBondType(), file=self.file)
          bType = bond.GetBondType()
          strType = str(bond.GetBondTypeAsDouble())
          if bType == Chem.BondType.SINGLE:
             bDir = bond.GetBondDir()
             if bDir in (Chem.BondDir.BEGINWEDGE, Chem.BondDir.BEGINDASH):
                # if the bond is "backwards", change the drawing direction:
                if bond.GetBeginAtom().GetChiralTag() in (Chem.ChiralType.CHI_TETRAHEDRAL_CW,Chem.ChiralType.CHI_TETRAHEDRAL_CCW):
                   strDir = ".0"
                else:
                   strDir = ".5"
                if bDir == Chem.BondDir.BEGINWEDGE:
                   strType = "4"+strDir
                elif bDir == Chem.BondDir.BEGINDASH:
                   strType = "5"+strDir

          self.file.write(str(self.index)+str(self.linenumber)+","+str(self.index)+","+strType+","+atom.GetSymbol()+","+str(atom.GetFormalCharge())+","+nbr.GetSymbol()+","+str(nbr.GetFormalCharge())+","+str(round(pos[0],2))+","+str(round(pos[1],2))+","+str(round(nbrPos[0],2))+","+str(round(nbrPos[1],2))+"\n") #save to file
         # file.close()
          self.linenumber=self.linenumber+1
          return

      def _drawLabel(self, label, pos, baseOffset, font, color=None, **kwargs):
          labelsize = super(MolDrawing, self)._drawLabel(label, pos, baseOffset, font, color, **kwargs)
          soup = BS(label, features="html.parser")
          charge = soup.find('sup')
          if charge is None:
              charge="0"
          else:
              if ("+" in charge.text) or ("-" in charge.text):
                 charge = charge.text
              else:
                  charge="0"
          if (len(str(charge)) == 1) and charge != "0":
              charge = str(charge)+"1"
          charge=charge.replace("+","")
          label = re.sub('<.*>','', label)
          if len(label) > 1 :
              label = re.sub('H','',label)
          self.file.write(str(self.index)+str(self.linenumber)+","+str(self.index)+",nobond,"+label+","+str(charge)+","+label+","+str(charge)+","+str(round(pos[0],2))+","+str(round(pos[1],2))+","+str(round(pos[0],2))+","+str(round(pos[1],2))+"\n")
          self.linenumber=self.linenumber+1
          return labelsize


      def newlineinfile(self,content="\n"):
          self.file.write(content)

      def setIndex(self, index=0):
          self.index=index
