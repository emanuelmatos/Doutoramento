#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd
import csv



# ## link para modelos

# https://huggingface.co/models

# ### config experiment

# In[8]:


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


# In[20]:


import sys 
base = pipeline
sys.path.append(base)

from config.configsite import *

#print(f"filesdir = {filesdir}") # new AT april 2025
print(f"pipeline = {pipeline}") # new AT april 2025


#inputdir=pipeline+"JOIN_MAPPER_REBEL/"+resultsdir
inputdir=pipeline+"NEW_INDOMAIN_NER_FLAIR/"+resultsdir   # AT 24 de maio 2025

print(f"input dir = {inputdir}") # new AT may 2025

outputdir=pipeline+"NEW_INDOMAIN_NER_FLAIR/"+resultsdir
print(f"output dir = {outputdir}") # new AT may 2025

os.makedirs(outputdir, exist_ok=True)


# In[21]:


# INPUT !! temos de garantir FORMATO similar
# dataset file

path =  inputdir
#filename= path+"results_class_top10.csv"  # versão do Emanuel
filename= path+"results_top10_adjst.csv"  # DUVIDA: Este está OK, verificado em 27 de maio com Emanuel

print(f"input file = {filename}") # new AT may 2025

# tag column to use
tag_column = "ner2"  # 1 class only   # é o OUTPUT do MAPPER
nline_column = 0

# diretory for dataset to be created
#dataset_dir = "./datawikiner_1class/"
#dataset_dir = "./datawikiner_2025_topNclass/"
#dataset_dir = "./data_bert_and_mapper_2025/"
dataset_dir = outputdir+"dataset_bert_and_mapper/"
print(f"output dataset dir = {dataset_dir}") # new AT may 2025

os.makedirs(dataset_dir, exist_ok=True)


# # process input (from file)

# In[22]:


##  below uses all data
#filename= ".\\wikiner_rebel_ner.txt"


# Inicialize um DataFrame vazio
df = pd.DataFrame()
    

# Inicialize uma lista para armazenar os dados
nlinha = []
palavras = []
ner1 = []
ner2 = []
ner3 = []
ner4 = []

num_linha = []
    
# Abra o arquivo CSV para leitura
with open(filename, 'r', encoding="utf8") as arquivo_csv:
    # Crie um objeto CSV reader
    #leitor_csv = csv.reader(arquivo_csv,delimiter=" ")
    
    


    
    # Percorra as linhas do arquivo CSV e adicione-as à lista de dados
    for i, linha in enumerate(arquivo_csv):
        if i == 0:
            continue
        
        linha = linha.rstrip()
        #if "\n" in linha:
        #print(f"|{linha}|")
            
        #if i> 185293:
        #    print(linha)
            
        partes = linha.split("\t")
        #if len(partes)>:
        #    print(f"partes = {partes[2]}")
        
        if len (partes) > 1:
            nlinha.append(partes[1])
            palavras.append(partes[2])
            ner1.append(partes[3])
            if len(partes)>2:
                ner2.append(partes[3])
            else:
                ner2.append("?")
            #ner3.append(partes[3])
            #ner4.append(partes[4])
        else:
            nlinha.append("")
            palavras.append("")
            ner1.append("")
            ner2.append("")
            #ner3.append("")
            #ner4.append("")
        
        num_linha.append(i+1)  # para começar em 1
        

# Crie um DataFrame a partir da lista de dados
df = pd.DataFrame({"n":num_linha,"word":palavras, 
                   "ner1":ner1, "ner2":ner2,"nlinha":nlinha
                   #"ner3":ner3 #, "ner4":ner4
                   })

# Exiba o DataFrame
#print(df)


# In[23]:


def save_to_file2(df, dir, file, tag="ner2"):
    filename = dir + file

    with open(filename, "w", encoding="utf8") as fout:
        last_nlinha = None

        for line in df.iterrows():
            dic = dict(line[1])
            current_nlinha = dic["nlinha"]

            # Se a nlinha mudou, escreve linha em branco (todas as colunas vazias)
            if last_nlinha is not None and current_nlinha != last_nlinha:
                fout.write("\t\t\n")  # 3 colunas: word, tag, nlinha

            str_line = f'{dic["word"]}\t{dic[tag]}\t{dic["nlinha"]}'
            fout.write(str_line + "\n")

            last_nlinha = current_nlinha

    print(f"✅ Arquivo salvo com separadores em: {filename}")


# In[15]:


limits=[0.7]
total = len(df)


# In[24]:


df.columns


# In[25]:


limits = [0.6, 0.1, 0.3]
limits = [0.8, 0.1, 0.1]


total = len(df)

columns_to_save = ['word', 'ner2', 'nlinha']

n_train = int(total * limits[0])
n_dev = int(total * limits[1])
n_test = total - n_train - n_dev  # Garante que soma dê 100%

train = df[:n_train]
print(train)
dev = df[n_train:n_train + n_dev]
test = df[n_train + n_dev:]

# Salvar arquivos
#save_to_file(train[columns_to_save], dataset_dir, "train.csv", tag_column)
#save_to_file(dev[columns_to_save], dataset_dir, "dev.csv", tag_column)
#save_to_file(test[columns_to_save], dataset_dir, "test.csv", tag_column)


# Verificar tamanhos
print(f"Tamanho do treino: {len(train)}")
print(f"Tamanho do dev: {len(dev)}")
print(f"Tamanho do teste: {len(test)}")


# In[26]:


save_to_file2(train, dataset_dir, "train.csv", tag_column)
save_to_file2(dev, dataset_dir, "dev.csv", tag_column)
save_to_file2(test, dataset_dir, "test.csv", tag_column)


# In[ ]:




