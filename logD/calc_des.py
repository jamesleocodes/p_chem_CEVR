"""
Author: Zaw
Calculating the descriptors
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from progressbar import ProgressBar
from rdkit.Chem import PandasTools, AllChem as Chem, Descriptors
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator

not_used_desc = ['MaxPartialCharge', 'MinPartialCharge', 'MaxAbsPartialCharge', 'MinAbsPartialCharge']
# Create a descriptor calculator for all RDKit descriptors except the ones above
desc_calc = MolecularDescriptorCalculator([x for x in [x[0] for x in Descriptors.descList] if x not in not_used_desc])


#load data_set
path = os.getcwd()
data_path = path+"/data/extract_data.csv"
col_list=['SMILES','LogP','LogD','PSA','a_pKa','b_pKa','Exp_RT']
df = pd.read_csv(data_path,usecols=col_list)


#df.drop(df[df['Exp_RT'] < 1].index,inplace=True)



# Convert SMILES to mol object
mol_obj = []


for element in df.SMILES:
    mol = Chem.MolFromSmiles(element)
    mol_obj.append(mol)


# Calculating the descriptors
print('\nDescriptors calculation started!!!')
pbar = ProgressBar() #initiate progress bar
descs = []
for mol in pbar(mol_obj):
#for mol in mol_obj:
    descs.append(desc_calc.CalcDescriptors(mol))
descs = np.array(descs)
df_descs = pd.DataFrame(descs,columns=([x for x in [x[0] for x in Descriptors.descList] if x not in not_used_desc]))
df_descs.to_csv(path+"/data/descriptor_1.csv")
print('Descriptors calculation is done and saved!!!')
