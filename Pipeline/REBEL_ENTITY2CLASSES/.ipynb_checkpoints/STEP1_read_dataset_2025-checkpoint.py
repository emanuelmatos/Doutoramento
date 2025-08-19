#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import sys
import tqdm


# In[2]:


import os

working_dir = os.getcwd()
print(working_dir)

parts = working_dir.split("/")
if parts[-1] == "pipeline":
    pipeline = working_dir
else:
    pipeline ="/".join(parts[:-1])
print(pipeline)


# In[3]:


#base = "/home/ematos/phd/pt/MAPPER_ENTITY2CLASSES/"
sys.path.append(pipeline)


#-----------------------------------------------------------------------------------------
from config.configsite import * #basedir,inputdir
 

inputdir=pipeline+"ENTITY_DETECTOR_BY_BERT/"+resultsdir
print(f"input dir = {inputdir}") # new AT april 2025


outputdir=pipeline+"REBEL_ENTITY2CLASSES/"+resultsdir
print(f"output dir = {outputdir}") # new AT april 2025
base = outputdir


#input_path = '/home/ematos/phd/pt/MAPPER_ENTITY2CLASSES/data/'
input_path = pipeline+"MAPPER_ENTITY2CLASSES/"

#input_dataset = '/home/ematos/phd/pt/DATASETS/INPUT_DATASET/'
input_dataset_processed = f"{pipeline}{dataset}_PROCESSED/"
input_dataset = f"{pipeline}{dataset}/"

 


lista_path = input_dataset_processed + "/files_list.csv"
print(f"output dir = {lista_path}")


# In[4]:


lista = pd.read_csv(lista_path)


# In[5]:


lista.loc[[0,len(lista)-1],]


# In[6]:


import numpy as np

def train_validate_test_split(df, train_percent=.7, validate_percent=.2, seed=123):
    np.random.seed(seed)
    perm = np.random.permutation(df.index)
    m = len(df.index)
    train_end = int(train_percent * m)
    validate_end = int(validate_percent * m) + train_end
    train = df.iloc[perm[:train_end]]
    validate = df.iloc[perm[train_end:validate_end]]
    test = df.iloc[perm[validate_end:]]
    return train, validate, test

train, validate, test = train_validate_test_split(lista)



# # create traind, validate and test

# # ?? Dúvida?  `TODO`
# Não deveria ser o mesmo split que já fez antes? 
# O de test inicial não devia ser nunca usado

# In[ ]:


os.makedirs(outputdir, exist_ok=True)


# In[7]:


# Sort DataFrame by filename
print("Creating train, validation and test sets")

train_sorted = train.sort_values(by="filename", ascending=True)
validate_sorted = validate.sort_values(by="filename", ascending=True)
test_sorted = test.sort_values(by="filename", ascending=True)


# Reset index
train_sorted = train_sorted.reset_index(drop=True)
validate_sorted = validate_sorted.reset_index(drop=True)
test_sorted = test_sorted.reset_index(drop=True)

file_train = base+'train_mapper_process.csv'
file_validate = base+'validate_mapper_process.csv'
file_test = base+'test_mapper_process.csv'

train_sorted.to_csv(file_train, index=False)
validate_sorted.to_csv(file_validate, index=False)
test_sorted.to_csv(file_test, index=False)



from sklearn.model_selection import train_test_split

train, test = train_test_split(df, test_size=0.3,random_state=123)

# In[ ]:


path_lista = base + "train_mapper_process.csv"
lista = pd.read_csv(path_lista)
lista.head()


# # TODO: ver se files_list_rebel_step.csv  deve ser diferente de files_list.csvs

# In[ ]:


#path_lista = base + "train_mapper_process.csv"
path_lista = f"{pipeline}{dataset}_PROCESSED/files_list_rebel_step.csv"
print(f"Processing from {path_lista}")


# In[ ]:


print(f"input dataset {input_dataset}")


# In[ ]:


def process_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    
    # Split text into sentences using punctuation (handling common sentence boundaries)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    # Create DataFrame with line numbers
    df = pd.DataFrame({'Line': range(1, len(sentences) + 1), 'Sentence': sentences})
    
    return df


# In[ ]:





# ### TRAIN

# In[ ]:


print(outputdir)


# In[ ]:


dir_train_output = f"{outputdir}DATASETS/TRAIN/"
print(f"dir 4 train dataset = {dir_train_output}")

os.makedirs(outputdir, exist_ok=True)

os.makedirs(f"{outputdir}DATASETS/TRAIN/", exist_ok=True)

os.makedirs(dir_train_output, exist_ok=True)


# In[ ]:


path_lista = base + "train_mapper_process.csv"
print(f"List for train = {path_lista}")
lista = pd.read_csv(path_lista)


# In[ ]:


from tqdm import tqdm
import re

for _, row in tqdm(lista.iterrows(), total=len(lista), desc="Processing files", unit="file"):
    filename = row.get('filename')
    basename = row.get('basename')


    file_path = input_dataset+filename # Change this to the actual file path
    try:
        df_sentences = process_text_file(file_path)
    except FileNotFoundError as e:
        print(f"*** Problems with {filename}, Skipping")
        continue
        
    filename = f"{outputdir}DATASETS/TRAIN/{filename}"
    print(f"\tSaving to {filename}")
    
    df_sentences.to_csv(filename, index=False, encoding='utf-8',sep='\t')
    #print(df_sentences)


# In[ ]:


dir_test_output = f"{outputdir}DATASETS/TEST/"
print(f"dir 4 test dataset = {dir_test_output}")
os.makedirs(outputdir, exist_ok=True)
os.makedirs(f"{outputdir}DATASETS/TEST/", exist_ok=True)
os.makedirs(dir_test_output, exist_ok=True)


# ### TEST

# In[ ]:


#### TEST
path_lista = base + "test_mapper_process.csv"
lista = pd.read_csv(path_lista)


for _, row in tqdm(lista.iterrows(), total=len(lista), desc="Processing files", unit="file"):
    filename = row.get('filename')
    basename = row.get('basename')

    file_path = input_dataset+filename # Change this to the actual file path
    try:
        df_sentences = process_text_file(file_path)
    except FileNotFoundError as e:
        print(f"*** Problems with {filename}, Skipping")
        continue
        
    filename = f"{outputdir}DATASETS/TEST/{filename}"
    print(f"\tSaving to {filename}")
    
    df_sentences.to_csv(filename, index=False, encoding='utf-8',sep='\t')
    
    


# ### VALIDATE

# In[ ]:


dir_validate_output = f"{outputdir}DATASETS/DEV/"
print(f"dir 4 validate (DEV) dataset = {dir_validate_output}")

os.makedirs(f"{outputdir}DATASETS/DEV/", exist_ok=True)
os.makedirs(dir_validate_output, exist_ok=True)


# In[ ]:


#### VALIDATE
path_lista = base + "validate_mapper_process.csv"
lista = pd.read_csv(path_lista)

for _, row in tqdm(lista.iterrows(), total=len(lista), desc="Processing files", unit="file"):
    filename = row.get('filename')
    basename = row.get('basename')


    file_path = input_dataset+filename # Change this to the actual file path
    try:
        df_sentences = process_text_file(file_path)
    except FileNotFoundError as e:
        print(f"*** Problems with {filename}, Skipping")
        continue
        
    filename = f"{outputdir}DATASETS/DEV/{filename}"
    print(f"\tSaving to {filename}")
    
    df_sentences.to_csv(filename, index=False, encoding='utf-8',sep='\t')
    


# In[ ]:





# In[ ]:




