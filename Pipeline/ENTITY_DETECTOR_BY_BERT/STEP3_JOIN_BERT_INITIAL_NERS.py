#!/usr/bin/env python
# coding: utf-8

# ## Inputb file = base_bert_2025.csv
# ## Output file = /home/ematos/phd/pt/ENTITY_DETECTOR_BY_BERT/OUTPUT_RESULTS/ner_to_flair.decision

# In[3]:


import os
import pandas as pd
import numpy as np
from pandas import DataFrame
import pandas as pd
import numpy as np
from tqdm import tqdm, trange
import pandas as pd
import csv  # Ensure csv is imported
from datetime import datetime
start_time = datetime.now()
import pandas as pd
import numpy as np
from tqdm import tqdm, trange
import sys


# In[4]:


debug=True


# In[5]:


working_dir = os.getcwd()
print(working_dir)

parts = working_dir.split("/")
#if parts[-1] == "pipeline":
if parts[-1].startswith("pipeline"): ### BUG SOLVED MAY 2025
    pipeline = working_dir
else:
    pipeline ="/".join(parts[:-1])
print(pipeline)


# In[6]:


base = pipeline
sys.path.append(base)

#-----------------------------------------------------------------------------------------
from config.configsite import * #basedir,inputdir

inputdir = pipeline+"INITIAL_NERS/"+resultsdir+"/"
inputdir2 = pipeline+"ENTITY_DETECTOR_BY_BERT/"+resultsdir+"/"

outputdir=pipeline+"ENTITY_DETECTOR_BY_BERT/"+resultsdir+"/"
print(f"output dir = {outputdir}") # new AT april 2025


# In[7]:


input_initial_ners = inputdir
filename1 = input_initial_ners+"decision_entity_bert_arrange.csv"

output_model = inputdir2 
filename2=output_model+f'output_bert_{dataset}.csv'

print(f"filename1 = {filename1}")
print(f"filename2 = {filename2}")


result_file = 'ner_to_flair.decision'


dat_ner =  pd.read_csv(filename1,sep='\t', encoding="utf8").ffill()
# TODO:  ter lista/array dat[]
# TODO: filename ser também uma lista
#dat_l = pd.read_csv(filename1, sep=",")
import pandas as pd
import csv

dat_bert = pd.read_csv(
    filename2,
    sep="\t",
    on_bad_lines='skip',  # Correct argument for newer Pandas versions
    quoting=csv.QUOTE_NONE,
    encoding="utf-8"
)


# In[8]:


print(dat_ner)


# In[9]:


# to avoid problems of " in names"
dat_bert.columns = dat_bert.columns.str.replace('"',"")

print(dat_bert.columns)


# In[10]:


# Step 1: Initialize BIO = Tag
dat_ner['BIO'] = dat_ner['Tag']

# Step 2: Prepare lookup table from dat_bert
dat_bert_lookup = dat_bert.set_index(['id_file', 'nword'])['BIO-BERT'].to_dict()

# Step 3: Update BIO only where current BIO is 'O' and match found
for idx, row in dat_ner.iterrows():
    file = row['FILE']
    pos = row['POSITION']
    if row['BIO'] == 'O':
        new_bio = dat_bert_lookup.get((file, pos), 'O')
        if new_bio != 'O':
            dat_ner.at[idx, 'BIO'] = new_bio


# In[11]:


dat_ner


# In[17]:


path_data = outputdir
file_data = path_data+f'output_bert_final_{dataset}.csv'
print(file_data)
dat_ner.to_csv(file_data,sep="\t",index=None)


# In[27]:


#path_data = "/home/ematos/phd/pt/MAPPER_ENTITY2CLASSES/INPUT_DATASET/"
#file_data = path_data+'output_bert_text_city.csv'
#dat_ner.to_csv(file_data,sep="\t",index=None)


# In[ ]:




