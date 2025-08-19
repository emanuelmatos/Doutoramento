#!/usr/bin/env python
# coding: utf-8

# # Ajuste TOP10

# In[1]:


# ====================================
# Imports and Warnings Setup
# ====================================
import pandas as pd
import csv
import warnings
import os
import torch
import flair
from flair.data import Corpus, Sentence
from flair.datasets import ColumnCorpus
from flair.embeddings import TransformerWordEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer

warnings.filterwarnings("ignore")  # Silence all warnings


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


# In[9]:


import sys 
base = pipeline
sys.path.append(base)

from config.configsite import *

#print(f"filesdir = {filesdir}") # new AT april 2025
print(f"pipeline = {pipeline}") # new AT april 2025


inputdir=pipeline+"JOIN_MAPPER_REBEL/"+resultsdir
print(f"input dir = {inputdir}") # new AT may 2025

outputdir=pipeline+"NEW_INDOMAIN_NER_FLAIR/"+resultsdir
print(f"output dir = {outputdir}") # new AT may 2025

os.makedirs(outputdir, exist_ok=True)


# # funçoes

# In[10]:


def compute_bio_tags(df):
    bio_tags = []

    prev_bio = 'O'
    prev_class = None
    prev_file = None
    prev_line = None

    for idx, row in df.iterrows():
        cur_file = row['file_id']
        cur_line = row['line']
        cur_class = row['MAPPER10']

        if cur_class == 'O':
            bio_tags.append('O')
            prev_bio = 'O'
            prev_class = None
        else:
            # Check if previous was part of the same class and sentence
            if (
                prev_file == cur_file and
                prev_line == cur_line and
                prev_bio.startswith(('B-', 'I-')) and
                prev_class == cur_class
            ):
                bio_tags.append(f'I-{cur_class}')
                prev_bio = f'I-{cur_class}'
            else:
                bio_tags.append(f'B-{cur_class}')
                prev_bio = f'B-{cur_class}'

            prev_class = cur_class

        prev_file = cur_file
        prev_line = cur_line

    return bio_tags


# # read data

# In[11]:


path = inputdir

filename = path + "results_mapper_rebel_top10_by_word.csv"  # Input file vem do JOIN_MAPPER_REBEL
#tag_column = "ner2"  # Column to use for tagging

df = pd.read_csv(filename,sep="\t")


# # apply processing

# In[12]:


# Apply to your DataFrame
df['BIO'] = compute_bio_tags(df)


# # save result

# In[13]:


path = outputdir
df.to_csv(path+'results_top10_adjst.csv',sep="\t", index=None) 


# ### ???

# In[14]:


df['BIO'] = compute_bio_tags(df)
df['x'] = df['WORD'].apply(lambda w: 1 if str(w).strip() == '.' else 0)


# In[ ]:


df


# In[ ]:




