#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python
# coding: utf-8

# Import necessary libraries
import requests
import re
import spacy
from spacy.tokens import Doc
from spacy import displacy
from pathlib import Path
import pandas as pd
import os
import pandas as pd
import sys
from tqdm import tqdm
from collections import Counter


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


#-----------------------------------------------------------------------------------
#base = "/home/ematos/phd/NER/process1/process_script/"
sys.path.append(pipeline)

# Configure directories
try:
    from config.configsite import * #filesdir, linguakitdir, inputdir, outputdir, basedir
except ImportError:
    print("Error: 'configsite' module not found. Ensure it is correctly configured.")
    sys.exit(1)

#-----------------------------------------------------------------------------------


# In[4]:


outputdir=pipeline+"INITIAL_NERS/"+resultsdir+"/"
print(f"output dir = {outputdir}") # new AT april 2025


# In[5]:


# Load SpaCy English model
nlp = spacy.load("en_core_web_sm")

# Define helper function to preprocess BIO annotations
def preprocess_bio_tags(entities):
    """Convert BIO tags to a consistent format."""
    processed_entities = []
    for item in entities:
        if 'U-' in item:
            item = item.replace('U-', 'B-')
        if 'L-' in item:
            item = item.replace('L-', 'I-')
        processed_entities.append(item)
    return processed_entities

# Load CSV file
filename = outputdir+"decision_final.csv"
df = pd.read_csv(filename,sep='\t')
df['linha'] = df['FILE']+'_'+df['LINE'].astype(str)


# In[6]:


#Step 1: Extract unique values
unique_values = df['linha'].unique()

# Step 2: Create a mapping from unique values to integers
value_to_int = {value: idx + 1 for idx, value in enumerate(unique_values)}

# Step 3: Apply the mapping to create a new column
df['mapped_value'] = df['linha'].map(value_to_int)

#df.columns = ['FILE','LINE','WORD', 'BIO', 'BIO1', 'BIO2', 'BIO3', 'WTA', 'ENTITY']
df['Sentence #'] = 'Sentence:'+df['mapped_value'].astype(str)

df['POS'] = 'X'

# Apply the preprocess_bio_tags function to specific columns
columns_to_process = ['BIO1', 'BIO2', 'BIO3','CONSENSUS','WTA']

for col in columns_to_process:
    df[col] = df[col].apply(lambda x: " ".join(preprocess_bio_tags(x.split())) if isinstance(x, str) else x)

df


# In[7]:


df["BIO"] = df["CONSENSUS"]
df.head(70)


# In[13]:


def update_bio(row):
    if row['BIO'] == 'O':
        for col in ('BIO1', 'BIO2', 'BIO3'):
            if row[col] != 'O' and pd.notna(row[col]):
                return row[col]  # Return the first non-'O' value
    return row['BIO']  # Keep original BIO if no changes
    
df['BIO'] = df.apply(update_bio, axis=1)
df


# In[14]:


def to_entity(label):
    if label == 'O':
        return label
    prefix, _ = label.split('-', 1)
    return f'{prefix}-ENTITY'

# Apply transformation to the BIO column
df['BIO'] = df['BIO'].apply(to_entity)


# In[15]:


df['BIO'].unique()


# In[16]:


def insert_blank_lines(df):
    """Insert a blank line whenever the LINE number changes."""
    new_rows = []
    prev_line = None  # Track the previous LINE value

    for i, row in df.iterrows():
        if prev_line is not None and row['LINE'] != prev_line:
            # Insert a blank row before the current row
            new_rows.append(pd.Series({col: '' for col in df.columns}))

        new_rows.append(row)
        prev_line = row['LINE']  # Update previous LINE value

    # Convert back to DataFrame
    return pd.DataFrame(new_rows, columns=df.columns)

# Apply the function
df = insert_blank_lines(df)
df


# In[17]:


df2 =df[['Sentence #','WORD','BIO','POS','LINE','POSITION','FILE']].copy()
df2.rename(columns={'WORD':'Word','BIO':'Tag'},inplace=True)


# In[18]:


df2


# In[19]:


df2.to_csv(f'{outputdir}/decision_entity_bert_arrange.csv', sep='\t')


# In[20]:


## ??? NEEDE FOR WAHT ??

# commented AT, april 2025
#df2.to_csv(f'/home/ematos/phd/pt/ENTITY_DETECTOR_BY_BERT/INPUT_DATASET/decision_entity_bert_arrange.csv', sep='\t')


# In[ ]:




