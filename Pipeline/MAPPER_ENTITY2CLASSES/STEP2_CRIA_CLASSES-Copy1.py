#!/usr/bin/env python
# coding: utf-8

# ## OUTPUT  em `NEW_INDOMAIN_NER_FLAIR  !!!`
# ### NEW_INDOMAIN_NER_FLAIR/dataset/results_class_top10.txt

# In[1]:


#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import math
import torch
from tqdm import tqdm
#from transformers import BertForTokenClassification, BertTokenizer
#from keras.preprocessing.sequence import pad_sequences
#from sklearn.model_selection import train_test_split
import codecs
import os
import sys
import pandas as pd
import numpy as np
import re
#---------------------------------------------------------------------------------
#sys.path.append(base)  # Add the base directory to the system path
#from configsite import filesdir, linguakitdir, inputdir, outputdir
#---------------------------------------------------------------------------


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
base = pipeline
sys.path.append(base)

#-----------------------------------------------------------------------------------------
from config.configsite import * #basedir,inputdir
 
 
inputdir=pipeline+"MAPPER_ENTITY2CLASSES/"+resultsdir
print(f"input dir = {inputdir}") # new AT april 2025


outputdir=pipeline+"MAPPER_ENTITY2CLASSES/"+resultsdir
print(f"output dir = {outputdir}") # new AT april 2025

inputllm=pipeline+"NEW_INDOMAIN_NER_LLM/"+resultsdir
# In[ ]:





# In[4]:


input_path = inputdir
input_file = "results_mapper_by_word.csv"

output_path = outputdir
output_file = "results_class_top10.csv"


# ## Ler resultados do Mapper

# In[5]:


import csv

# Resultados e Classificação do Mapper
print(f"Reading from {input_path+input_file}...")
#
#d_mapper = pd.read_csv(input_path+input_file, sep="\t", engine='python')

# NEW version AT 8 junho 2025
d_mapper = pd.read_csv(input_path+input_file, sep="\t",quoting=csv.QUOTE_NONE)

print(f"Done.")
d1 = d_mapper[['id_file','Line','WORD','NER','MAPPER']].copy()
#d1.rename(columns={"WORD": "WORD","result":"class"}, inplace=True)

print(d_mapper.head())


# In[6]:


d1['MAPPER'] = d1['MAPPER'].str.replace(r'^[BI]-', '', regex=True)
print(d1.head())


# In[7]:


d1['MAPPER'] = d1['MAPPER'].str.replace(r'^Q\d+', 'O', regex=True)
d1


# In[8]:


import pandas as pd

# Assume df is your original DataFrame
line_change = d1['Line'] != d1['Line'].shift()

blank_row = {col: None for col in d1.columns}
new_rows = []

for idx, row in d1.iterrows():
    if idx > 0 and line_change.iloc[idx]:
        new_rows.append(blank_row.copy())  # Add blank row before line change
    new_rows.append(row.to_dict())  # Convert row to dict

# Now convert to DataFrame
df_with_blanks = pd.DataFrame(new_rows)

# Optional: reset index
df_with_blanks.reset_index(drop=True, inplace=True)


# In[9]:


file_new =output_path+"results_mapper_by_word_adjust.csv"
print(f"Saving to {file_new}...")
df_with_blanks.to_csv(file_new, sep = "\t", index = False)
print("Done.")


# In[10]:


import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

df = d1

# Step 1: Filter rows where class is ENT
ent_df = df

# Step 2: Calculate word distribution for ENT
word_distribution = ent_df['WORD'].value_counts().reset_index()
word_distribution.columns = ['WORD', 'count']
word_distribution['percentage'] = (word_distribution['count'] / word_distribution['count'].sum()) * 100

# Sort the word distribution by percentage in descending order
word_distribution = word_distribution.sort_values(by='percentage', ascending=False)

# Step 3: Save the ENT word distribution table to a CSV file
word_distribution.to_csv('ent_word_distribution.csv', index=False)
print("ENT word distribution table saved as 'ent_word_distribution.csv'")

# Step 1: Create a table with all classes
class_counts_all = df['MAPPER'].value_counts().reset_index()
class_counts_all.columns = ['MAPPER', 'count']
class_counts_all['percentage'] = (class_counts_all['count'] / class_counts_all['count'].sum()) * 100

# Sort the table by percentage in descending order
class_counts_all = class_counts_all.sort_values(by='percentage', ascending=False)

# Step 2: Save the full table to a CSV file
class_counts_all.to_csv('class_counts_table.csv', index=False)
print("Table saved as 'class_counts_table.csv'")

# Step 3: Create filtered dataframe for graphs (exclude 'O' and 'página de desambiguação da Wikimedia')
filtered_df = df[
    (df['MAPPER'] != 'O') & 
    (df['MAPPER'] != 'B-O') & 
    (df['MAPPER'] != 'I-O') & 
    (df['MAPPER'] != 'página de desambiguação da Wikimedia') & 
    (df['MAPPER'] != 'PÁGINA_DE_DESAMBIGUAÇÃO_DA_WIKIMEDIA') & 
    (df['MAPPER'] != 'B-PÁGINA_DE_DESAMBIGUAÇÃO_DA_WIKIMEDIA') & 
    (~df['MAPPER'].str.match(r'^Q\d+', na=False)) &
    (df['MAPPER'] != 'caractere Unicode') & 
    (df['MAPPER'] != 'CARACTERE_UNICODE') &
    (df['MAPPER'] != 'caractere') &
(df['MAPPER'] != 'CARACTERE') &
(df['MAPPER'] != 'GENE') &
(df['MAPPER'] != 'ANO') &
(df['MAPPER'] != 'UNIDADE_DE_MASSA') &
    (df['MAPPER'] != 'pontuação') &
    (df['MAPPER'] != 'PONTUAÇÃO') &
(df['MAPPER'] != 'ÁLBUM') &
    (df['MAPPER'] != 'preposição') &
    (df['MAPPER'] != 'entrada no dicionário')
    ]

class_counts_filtered = filtered_df['MAPPER'].value_counts().reset_index()
class_counts_filtered.columns = ['MAPPER', 'count']
class_counts_filtered['percentage'] = (class_counts_filtered['count'] / class_counts_filtered['count'].sum()) * 100

# Sort the filtered dataframe by percentage in descending order
class_counts_filtered = class_counts_filtered.sort_values(by='percentage', ascending=False)

# Step 4: Save filtered top 1, top 5, top 10, and top 20 classes to CSV files
class_counts_filtered.head(1).to_csv(output_path+'top1.csv', index=False)
class_counts_filtered.head(5).to_csv(output_path+'top5.csv', index=False)
class_counts_filtered.head(10).to_csv(output_path+'top10.csv', index=False)
class_counts_filtered.head(20).to_csv(output_path+'top20.csv', index=False)

print("Top 1, 5, 10, and 20 classes saved as 'top1.csv', 'top5.csv', 'top10.csv', and 'top20.csv'")

# Step 5: Save the graphs to a PDF file
with PdfPages(output_path+'class_distribution_graphs.pdf') as pdf:
    # Function to plot horizontal bar graphs in decreasing order
    def plot_horizontal_bar(data, title, top_n=None):
        if top_n:
            data = data.head(top_n)
        # Ensure the data is sorted in descending order
        data = data.sort_values(by='percentage', ascending=True)  # Ascending for horizontal bars (highest at top)
        plt.figure(figsize=(10, 6))
        bars = plt.barh(data['MAPPER'], data['percentage'], color='skyblue')
        plt.xlabel('Percentage')
        plt.ylabel('MAPPER')
        plt.title(title)
        # Add percentage labels to the bars
        for bar in bars:
            width = bar.get_width()
            plt.text(width, bar.get_y() + bar.get_height() / 2, f'{width:.1f}%', 
                     va='center', ha='left', fontsize=10)
        plt.tight_layout()
        pdf.savefig()  # Save the current figure to the PDF
        plt.close()

    # Plot graphs for top 5, top 10, top 15, and all remaining classes
    plot_horizontal_bar(class_counts_filtered, 'Top 5 Classes by Percentage (Excluding "O" and "página de desambiguação da Wikimedia")', top_n=5)
    plot_horizontal_bar(class_counts_filtered, 'Top 10 Classes by Percentage (Excluding "O" and "página de desambiguação da Wikimedia")', top_n=10)
    plot_horizontal_bar(class_counts_filtered, 'Top 15 Classes by Percentage (Excluding "O" and "página de desambiguação da Wikimedia")', top_n=15)
    plot_horizontal_bar(class_counts_filtered, 'All Classes by Percentage (Excluding "O" and "página de desambiguação da Wikimedia")')

print("Graphs saved as 'class_distribution_graphs.pdf'")


# In[ ]:





# In[ ]:





# In[11]:


d10 = class_counts_filtered.head(10)
dat = df


# In[12]:


# Merge d1 and d2 on the 'WORD' column
merged_d10 = pd.merge(d1, d10, on='MAPPER', how='left')
d10 = merged_d10.dropna()
d10.rename(columns={'MAPPER':'MAPPER10'},inplace=True)
d10.to_csv(f"{inputllm}/results_top10_adjst.csv")


# # ??? onde se cria ner_bert.decision
# 
# 

# path_data = "/home/ematos/phd/pt/MAPPER_ENTITY2CLASSES/data/"
# file_data = 'ner_bert.decision'
# 
# # Resultados Dos NER's base + BERT
# d_bert = pd.read_csv(path_data+file_data, sep="\t", engine='python')
# d2 = d_bert[['Line','WORD']].copy()
# 
# merged_d10 = pd.merge(d2, d10[['WORD','class']], on='WORD', how='left').fillna('O')
# #d10_f = merged_d10['class'].fillna('O')
# #d10_f
# dfim = merged_d10.drop_duplicates()

# dfim[['WORD','class','Line']].to_csv(output_path+output_file, sep= "\t",index=False)

# In[ ]:




