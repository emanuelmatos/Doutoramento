#!/usr/bin/env python
# coding: utf-8

# ## OUTPUT  em `NEW_INDOMAIN_NER_FLAIR  !!!`
# ### NEW_INDOMAIN_NER_FLAIR/dataset/results_class_top10.txt

# In[1]:


import pandas as pd
import numpy as np
import math
import torch
from tqdm import tqdm
from transformers import BertForTokenClassification, BertTokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import codecs
import os
import sys
import pandas as pd
import numpy as np
import re


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


# In[4]:


#---------------------------------------------------------------------------------
#sys.path.append(base)  # Add the base directory to the system path
#from configsite import filesdir, linguakitdir, inputdir, outputdir
#---------------------------------------------------------------------------



#input_path = "/home/ematos/phd/pt/JOIN_MAPPER_REBEL/"
input_path = pipeline+"JOIN_MAPPER_REBEL/"+resultsdir
print(f"Input from {input_path}")

input_file = "resultados_mapper_rebel.csv"

#output_path = "/home/ematos/phd/pt/JOIN_MAPPER_REBEL/"
output_path = input_path
print(f"Output to {output_path}")


output_file = "results_class_top10.csv"


# ## Ler resultados do Mapper

# In[5]:


# Resultados e Classificação do Mapper
d_mapper = pd.read_csv(input_path+input_file, sep="\t", engine='python')
d1 = d_mapper[['file_id','line','WORD','NER','final_result']].copy()
#d1.rename(columns={"WORD": "WORD","result":"class"}, inplace=True)

d_mapper.head(3)


# In[6]:


d1['final_result'] = d1['final_result'].str.replace(r'^Q\d+', 'O', regex=True)
d1.sample(4)


# In[7]:


# Assume df is your original DataFrame
line_change = d1['line'] != d1['line'].shift()

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


# In[8]:


file_new =output_path+"results_mapper_by_word_adjust.csv"

print(f"Saving to {file_new}")
df_with_blanks.to_csv(file_new, sep = "\t", index = False)


# In[9]:


import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

df = d1

df['MAPPER'] = df['final_result']

# Step 1: Filter rows where class is ENT
ent_df = df

# Step 2: Calculate word distribution for ENT
word_distribution = ent_df['WORD'].value_counts().reset_index()
word_distribution.columns = ['WORD', 'count']
word_distribution['percentage'] = (word_distribution['count'] / word_distribution['count'].sum()) * 100

# Sort the word distribution by percentage in descending order
word_distribution = word_distribution.sort_values(by='percentage', ascending=False)

# Step 3: Save the ENT word distribution table to a CSV file
word_distribution.to_csv(output_path+'ent_word_distribution.csv', index=False)
print("ENT word distribution table saved as 'ent_word_distribution.csv'")

# Step 1: Create a table with all classes
class_counts_all = df['MAPPER'].value_counts().reset_index()
class_counts_all.columns = ['MAPPER', 'count']
class_counts_all['percentage'] = (class_counts_all['count'] / class_counts_all['count'].sum()) * 100

# Sort the table by percentage in descending order
class_counts_all = class_counts_all.sort_values(by='percentage', ascending=False)

# Step 2: Save the full table to a CSV file
aux = output_path+'class_counts_table.csv'
class_counts_all.to_csv(aux, index=False)
print(f"Table saved as {aux}")

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
    (df['MAPPER'] != 'ANO') &
    (df['MAPPER'] != 'pontuação') &
    (df['MAPPER'] != 'preposição') &

    (df['MAPPER'] != 'GENE') &
(df['MAPPER'] != 'ALBUM') &
(df['MAPPER'] != 'ÁLBUM') &
(df['MAPPER'] != 'UNIDADE_DE_MASSA') &
(df['MAPPER'] != 'ANO_BISSEXTO') &
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


# In[10]:


df.sample(4)


# In[11]:


output_file = 'results_mapper_rebel_by_word.csv'
path_data = output_path

aux = path_data+output_file
print(f"Creating {aux}")
df[['file_id','line','WORD','final_result']].to_csv(aux, sep= "\t",index=False)


# In[12]:


d10=class_counts_filtered.head(10)
d10


# In[13]:


d1['MAPPER10']='O'


# In[14]:


d10 = class_counts_filtered.head(10)


# In[20]:


# Substitui com valor da coluna MAPPER se ele estiver entre os 10 do d10
top_10_labels = set(d10['MAPPER'])
d1.loc[d1['MAPPER'].isin(top_10_labels), 'MAPPER10'] = d1['MAPPER']
d1.sample(4)


# In[21]:


d1['MAPPER10'].unique()


# In[22]:


# Garante que cada valor termine com ".txt" (sem duplicar .txt caso já tenha)
d1['file_id'] = d1['file_id'].astype(str).apply(lambda x: x if x.endswith('.txt') else x + '.txt')


# In[24]:


output_file = 'results_mapper_rebel_top10_by_word.csv'
path_data = output_path

aux = path_data+output_file
print(f"Creating {aux}")

d1[['file_id','line','WORD','MAPPER10']].to_csv(aux, sep= "\t",index=False)

inputllm=pipeline+"NEW_INDOMAIN_NER_LLM/"+resultsdir

aux2 = inputllm+'results_top10_adjst.csv'
d1[['file_id','line','WORD','MAPPER10']].to_csv(aux2, sep= "\t",index=False)

# path_data = "/home/ematos/phd/pt/NEW_INDOMAIN_NER_FLAIR/INPUT_DATASET/"
# d1[['file_id','line','WORD','MAPPER10']].to_csv(path_data+output_file, sep= "\t",index=False)
# In[ ]:




