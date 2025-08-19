#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd
import sys
from tqdm import tqdm


# In[2]:


working_dir = os.getcwd()
print(working_dir)

parts = working_dir.split("/")
#if parts[-1] == "pipeline":
if parts[-1].startswith("pipeline"): ### BUG SOLVED MAY 2025
    pipeline = working_dir
else:
    pipeline ="/".join(parts[:-1])
print(pipeline)


# In[3]:


base = pipeline
sys.path.append(base)

#-----------------------------------------------------------------------------------------
from config.configsite import * #basedir,inputdir


# In[4]:


output_dir = f'{pipeline}/{dataset}_PROCESSED/{resultsdir}'

import os

os.makedirs(output_dir, exist_ok=True)

print(f"Output to {output_dir}")


# In[5]:


#base = "/home/ematos/phd/NER/process1/process_script/"
 
 
# Nome da lista de arquivos a processar
nome_lista = f'{pipeline}/{dataset}_PROCESSED/files_list.csv'
lista_path = nome_lista

lista = pd.read_csv(lista_path)
lista.head(4)


# In[6]:


# Initialize a list to store all data with sequential numbering
all_data = []
#line_number = 1  # Global line counter

# Iterate through each file in the list
for _, row in tqdm(lista.iterrows(), total=len(lista), desc="Processing files", unit="file"):
    filename = row.get("filename")
    basename = row.get("basename")
    line_number = 1  # restart for each file
    
    file_path = os.path.join(inputdir, f"{basename}.txt")
    
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"Warning: File {file_path} not found. Skipping...")
        continue
    
    # Read file line by line and store with sequential numbering
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            stripped_line = line.strip()
            if stripped_line:  # Skip blank lines
                all_data.append({
                    "line_number": line_number,
                    "filename": filename,
                    "text": stripped_line
                })
            line_number += 1


# In[7]:


# Convert to DataFrame
final_df = pd.DataFrame(all_data)

# Save concatenated output
output_file = os.path.join(output_dir, "concatenated_output_sentences_v2.csv")
print(f"Saving to {output_file}")
final_df.to_csv(output_file, index=False, encoding="utf-8")

print(f"Processed {len(all_data)} non-blank lines. Output saved to {output_file}.")


# In[ ]:




