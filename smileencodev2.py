from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from karateclub import Graph2Vec
from rdkit import Chem
from rdkit.Chem import AllChem
import urllib.request as urllib # https://docs.python.org/3/library/urllib.request.html
import pandas as pd
import pysmiles
import xgboost as xgb
import numpy as np
import networkx as nx
import time
import pickle



class ProcessSmiles:
	"""
	Generate the Smile encoded vector
	"""
	def __init__(self,smiles):
		"""
		Initialise your class...
		"""
		self.smiles = smiles

	# convert SMILES to graph with rdkit
	def mol_to_vec(self):
		"""
		Convert molecule string to graph network
		"""
		mol = Chem.MolFromSmiles(self.smiles.strip())
		fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 1024)
		bitstring = fp.ToBitString()
		intmap = map(int, bitstring)
		return np.array(list(intmap))
    
if __name__ == "__main__":
	# Load the coronavirus data 
	url = "https://raw.githubusercontent.com/yangkevin2/coronavirus_data/master/data/amu_sars_cov_2_in_vitro.csv"
	raw_data = urllib.urlopen(url)

	# load the CSV file as a Pandas Dataframe
	df = pd.read_csv(raw_data)
	print(df)
	# Split the ones from the zeroes evenly
	array_neg = df[df['fda'] == 0.0].values[:, 0:2] # Gather all the negative examples
	array_pos = df[df['fda'] == 1.0].values[:, 0:2] # Gather all the positive examples

	array_neg = np.resize(array_neg, (1396, 2))

	print(len(array_neg), len(array_pos), len(array_neg) + len(array_pos))

	# Homogenously mix the positive and negative examples
	spacing = int(len(array_neg) / len(array_pos))
	indices = np.array([*range(len(array_pos))]) * spacing
	array = np.insert(array_neg, indices, array_pos, axis=0)
	print(len(array_neg), len(array_pos), len(array))

	X = array[:,0]

	y = array[:,1]
	print(X) # COL 1: chemical makeup of a compound
	print(y) # COL 2: binary value indicating prediction of whether or not this molecule exhibits properties that
	# Convert SMILES Strings to numerical vectors so that XGBoost can process them in such a way that trends can be established
	# when molecules have similar structure.

	X_enc = []
	for smiles in X:
		#mol = pysmiles.read_smiles(smiles, explicit_hydrogen=True) #doall
		mol_graph_encoder = ProcessSmiles(smiles)
		mol_vec = mol_graph_encoder.mol_to_vec()
		X_enc.append(mol_vec)
	df_HDvec = pd.DataFrame(X_enc)
	df_HDvec['class'] = y
	df_HDvec.to_csv('molecule_vector.csv',index=False)

	with open('Molecule_enc', 'wb') as fp:
		pickle.dump(X_enc, fp)
        
	with open('Molecule_class', 'wb') as fp1:
		pickle.dump(y, fp1)
        
	with open('Molecule_df', 'wb') as fp2:
		pickle.dump(df_HDvec, fp2)
