"""
Author: Zaw
Scraping the P-Chem properties for the molecules in the smrt dataset
"""
import pandas as pd
import requests
import re
import os
import csv
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import Draw
from chembl_webresource_client.new_client import new_client
from tqdm import trange, tqdm
from time import sleep
from rdkit.Chem import MolFromSmiles
from rdkit.Chem.MolStandardize.rdMolStandardize import ChargeParent
from rdkit import RDLogger
__author__ ='zaw'
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)

# Reading smrt dataset
path = os.getcwd()
dirname = os.path.dirname(path)
print(dirname)
data_path = dirname+"/p_chem/data/SMRT_dataset.csv"
smrt_dataset = pd.read_csv(data_path, sep=';')

# Convert from InChi notation to SMILES
inchi_code_list = smrt_dataset.inchi
rt_list = smrt_dataset.rt
new_rt_list = []
smiles_code_list = []
for (a,b) in zip(inchi_code_list,rt_list):
        try:
            d =Chem.MolToSmiles(Chem.MolFromInchi(a))
            smiles_code_list.append(d)
            new_rt_list.append(b)
        except:
            pass

# smrt object list

new_smrt_dataset_list = {'rt_in_sec':new_rt_list,'SMILES':smiles_code_list,}
new_smrt_dataset = pd.DataFrame(new_smrt_dataset_list)


## Searching the ChEMBLE for molecules in database

list_molecule = new_smrt_dataset.SMILES[0:10000]
exp_rt = new_smrt_dataset.rt_in_sec[0:10000]
logp_list =[]
logd_list = []
psa_list = []
apka_list = []
bpka_list =[]
toxicity_list =[]
new_smiles =[]
rt_lst = []
count = 0
for (smiles,rt) in zip(list_molecule,exp_rt):

    try:
        molecule = new_client.molecule
        records = molecule.get([smiles])
        logp_list.append(records[0]['molecule_properties']['cx_logp'])
        logd_list.append(records[0]['molecule_properties']['cx_logd'])
        psa_list.append(records[0]['molecule_properties']['psa'])
        apka_list.append(records[0]['molecule_properties']['cx_most_apka'])
        bpka_list.append(records[0]['molecule_properties']['cx_most_bpka'])
        toxicity_list.append(records[0]['withdrawn_class'])
        new_smiles.append(smiles)
        rt_lst.append(rt)
    except:
            pass
    count += 1
    for j in tqdm(range(100)):
        sleep(0.001)
    print('Number of molecule: '+str(count)+' have been searched.')

#data = {'SMILES':new_smiles,'LogP':logp_list}
data={'SMILES':new_smiles,'LogP':logp_list,'LogD':logd_list,'PSA':psa_list,'a_pKa':apka_list,'b_pKa':bpka_list,'Toxicity':toxicity_list,'Exp_RT':rt_lst}
df = pd.DataFrame(data)

# Save to csv in order not to search in ChEMBLE
df.to_csv(dirname+"/p_chem/data/p_chem.csv")
