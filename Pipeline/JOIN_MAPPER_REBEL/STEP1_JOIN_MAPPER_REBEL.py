#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


import os

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


import sys
sys.path.append(pipeline)

from config.configsite import * #basedir,inputdir
outputdir=pipeline+"JOIN_MAPPER_REBEL/"+resultsdir
print(f"output dir = {outputdir}") # new AT april 2025

# ORIGINAL do Emanuel
#rebel = pd.read_csv('/home/ematos/phd/pt/REBEL_ENTITY2CLASSES/RESULTS/results_rebel.csv', sep = '\t')
#mapper = pd.read_csv('/home/ematos/phd/pt/MAPPER_ENTITY2CLASSES/OUTPUT_RESULTS/results_mapper_by_word.csv', sep = '\t')
# In[4]:


rebel_dir = f"{pipeline}REBEL_ENTITY2CLASSES/{resultsdir}"
print(f"REBEL dir = {rebel_dir}")
mapper_dir = f"{pipeline}MAPPER_ENTITY2CLASSES/{resultsdir}"
print(f"Mapper dir = {mapper_dir}")


# # !!! atenção é preciso eliminar 

# In[10]:


debug = False
#debug = True
if debug:
    rebel_emanuel = pd.read_csv('/home/ematos/phd/pt/REBEL_ENTITY2CLASSES/RESULTS/results_rebel.csv', sep = '\t')
    rebel_emanuel.head()
    rebel = rebel_emanuel
else:
    aux = rebel_dir+'results_rebel.csv'
    print(f"Reading from {aux}")
    rebel = pd.read_csv(aux, sep = '\t')    


# # Mapper results

# In[11]:


mapper = pd.read_csv(mapper_dir+'results_mapper_by_word.csv', sep = '\t', quoting=3)
print(mapper.columns)
print(mapper.tail())


# In[12]:


rebel.columns


# In[13]:


mapper.rename(columns={'id_file':'file_id','Line':'line'},inplace=True)


# # Expand Rebel results 

# In[14]:


df = rebel
rows = []
for idx, row in df.iterrows():
    words = row['base'].split()
    for word in words:
        new_row = {
            "line": row['line'],
            "mapper": row['mapper'],
            "base": word,
            "result": row['result'],
            "file_id": row['file_id']
        }
        rows.append(new_row)

# Create the new dataframe
df_expanded = pd.DataFrame(rows)

# Display result
print(df_expanded)


# In[15]:


mapper_files = mapper['file_id'].drop_duplicates().sort_values().reset_index(drop=True)


# In[16]:


print(mapper_files)


# In[17]:


rebel_files = rebel['file_id'].drop_duplicates().sort_values().reset_index(drop=True)


# In[18]:


rebel_files


# In[19]:


# Ensure all strings (if needed) to avoid mismatches due to types
df_expanded['line'] = df_expanded['line'].astype(str)
mapper['line'] = mapper['line'].astype(str)
df_expanded['base'] = df_expanded['base'].astype(str)
mapper['WORD'] = mapper['WORD'].astype(str)
df_expanded['file_id'] = df_expanded['file_id'].astype(str)
mapper['file_id'] = mapper['file_id'].astype(str)




##### REVER OS B- e I- 
#####################################################REVER   
# Merge on line, base==WORD, and file_id
merged = pd.merge(
    df_expanded,
    mapper,
    left_on=['line', 'base', 'file_id'],
    right_on=['line', 'WORD', 'file_id'],
    how='right'
)
#####################################################REVER  
# Optional: drop duplicate columns like 'WORD' or 'MAPPER' if not needed
# merged = merged.drop(columns=['WORD'])

# View result
print(merged.head())


# In[20]:


merged


# In[21]:


# Step 1: Define final_result logic
merged['final_result'] = merged.apply(
    lambda row: row['result'] if row['MAPPER'] in ['O', 'B-PÁGINA_DE_DESAMBIGUAÇÃO_DA_WIKIMEDIA'] and pd.notna(row['result']) else (
        "O" if row['MAPPER'] in ['O', 'B-PÁGINA_DE_DESAMBIGUAÇÃO_DA_WIKIMEDIA'] else row['MAPPER']
    ),
    axis=1
)

# Step 2: Format final_result
merged['final_result'] = (
    merged['final_result']
    .str.upper()                     # UPPERCASE
    .str.replace(' ', '_')          # Replace spaces with underscores
    .str.replace(r'^[BI]-', '', regex=True)  # Remove B- or I- at the beginning
)

# Optional: check result
print(merged.head())


# In[22]:


os.makedirs(outputdir, exist_ok=True)


# In[23]:


filename=f"{outputdir}resultados_mapper_rebel.csv"
print(f"Saving to {filename}")

merged.to_csv(filename, sep = '\t', index = None)


# In[24]:


#merged.to_csv('/home/ematos/phd/pt/NEW_INDOMAIN_NER_FLAIR/INPUT_DATASET/resultados_mapper_rebel.csv', sep = '\t', index = None)

