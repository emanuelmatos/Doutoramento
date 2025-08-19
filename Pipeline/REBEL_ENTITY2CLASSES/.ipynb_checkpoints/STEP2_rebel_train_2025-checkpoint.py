#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import pandas as pd
import numpy as np
from tqdm import tqdm, trange
import os
import codecs
import warnings
warnings.filterwarnings("ignore")
import os 
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


#base = "/home/ematos/phd/pt/MAPPER_ENTITY2CLASSES/"
sys.path.append(pipeline)

from config.configsite import * #basedir,inputdir

outputdir=pipeline+"REBEL_ENTITY2CLASSES/"+resultsdir
print(f"output dir = {outputdir}") # new AT april 2025
base = outputdir

input_path = f"{pipeline}REBEL_ENTITY2CLASSES/{resultsdir}/DATASETS/"


#input_dataset = '/home/ematos/phd/pt/DATASETS/INPUT_DATASET/'

#output_dir = '/home/ematos/phd/pt/MAPPER_ENTITY2CLASSES/OUPUT_RESULTS/'
output_dir = f"{pipeline}REBEL_ENTITY2CLASSES/{resultsdir}RESULTS/"
print(f"output_dir = {output_dir}")

#path_lista = base + "train_mapper_process.csv"


# In[4]:


os.makedirs(output_dir, exist_ok=True)


# In[5]:


path_lista = base + "train_mapper_process.csv"
print(f"List for train = {path_lista}")
lista = pd.read_csv(path_lista)


# In[6]:


lista = pd.read_csv(path_lista)


# In[7]:


lista.head(3)


# In[8]:


from transformers import pipeline
    # Function to parse the generated text and extract the triplets
def extract_triplets(text):
    triplets = []
    subject, relation, object_ = "", "", ""
    current = None  # Tracks whether we are processing subject, relation, or object

    for token in text.split():
        if token == "<triplet>":
            if subject and relation and object_:
                triplets.append((subject.strip(), relation.strip(), object_.strip()))
            subject, relation, object_ = "", "", ""
            current = "t"
        elif token == "<subj>":
            if subject and relation and object_:
                triplets.append((subject.strip(), relation.strip(), object_.strip()))
            subject, relation, object_ = "", "", ""
            current = "s"
        elif token == "<obj>":
            current = "o"
        else:
            if current == "t":
                subject += " " + token
            elif current == "s":
                object_ += " " + token
            elif current == "o":
                relation += " " + token

    if subject and relation and object_:  # Append the last triplet if not empty
        triplets.append((subject.strip(), relation.strip(), object_.strip()))

    return triplets


# # TRAIN

# In[9]:


#path_lista = base + "train_mapper_process.csv"
#
#lista = pd.read_csv(path_lista)


import re
from transformers import pipeline
from tqdm import tqdm
import pandas as pd
from argostranslate.translate import translate as argos_translate



# Inicializar o pipeline REBEL uma vez
triplet_extractor = pipeline('text2text-generation', 
                             model='Babelscape/rebel-large', tokenizer='Babelscape/rebel-large')


# In[ ]:


print(output_dir)


# In[ ]:


os.makedirs(output_dir+"TRAIN", exist_ok=True)


# In[10]:


#### TRAIN
for _, row in tqdm(lista.iterrows(), total=len(lista), desc="Processing files", unit="file"):

    #output_dir = "/home/ematos/phd/pt/MAPPER_ENTITY2CLASSES/OUTPUT_RESULTS/"
    #output_dir = outputdir
    filewiki = row.get('filename')
    basename = row.get('basename')

    resp = f'{output_dir}TRAIN/{filewiki}'

    if not (os.path.isfile(resp) and os.path.getsize(resp) > 0):

        aux = f"{input_path}TRAIN/{filewiki}"
        print(f"Reading from {aux}")
    
        df = pd.read_csv(aux, sep='\t')
        df2 = df.sort_values('Line')



    
        output_rebel = open(resp, 'w', encoding="utf8")
        output_rebel.write(f"line_number\tmap\n")
    
        for line in df2.iterrows():
            word = line[1][1]
            lin = line[1][0]
    
            word = str(word) if pd.notna(word) else ""
            frase = word.strip().split("\t")
            test_sentence = ' '.join(map(str, frase))
    
            # Extração REBEL
            extracted_text = triplet_extractor.tokenizer.batch_decode(
                [triplet_extractor(test_sentence, return_tensors=True, return_text=False)[0]["generated_token_ids"]]
            )[0]
    
            # Traduzir apenas o conteúdo das tags
            def traduzir_tripletos(texto):
                def traduzir_grupo(match):
                    partes = match.group(0)
    
                    # Extrai os elementos
                    subj = re.search(r"<subj>(.*?)<obj>", partes)
                    obj = re.search(r"<obj>(.*?)($|<triplet>)", partes)
                    triplet = re.search(r"<triplet>(.*?)<subj>", partes)
    
                    s = subj.group(1).strip() if subj else ""
                    o = obj.group(1).strip() if obj else ""
                    t = triplet.group(1).strip() if triplet else ""
    
                    # Traduz individualmente
                    try:
                        t_pt = argos_translate(t, "en", "pt") if t else ""
                        s_pt = argos_translate(s, "en", "pt") if s else ""
                        o_pt = argos_translate(o, "en", "pt") if o else ""
                    except Exception as e:
                        print(f"[Falha tradução linha {lin}]: {e}")
                        return partes  # fallback: retorna como está
    
                    return f"<triplet> {t_pt} <subj> {s_pt} <obj> {o_pt}"
    
                # Traduz todos os grupos
                novo_texto = re.sub(r"<triplet>.*?<subj>.*?<obj>.*?(?=<triplet>|</s>)", traduzir_grupo, texto)
                return novo_texto
    
            try:
                texto_pt_formatado = traduzir_tripletos(extracted_text)
                output_rebel.write(f"{lin}\t{texto_pt_formatado}\n")
            except Exception as e:
                print(f"[Erro geral linha {lin}]: {e}")
                output_rebel.write(f"{lin}\t{extracted_text}\n")  # fallback
    
        output_rebel.close()
    else:
        print("Processing results already exist, skipping")


# # TEST

# In[ ]:


os.makedirs(output_dir+"TEST", exist_ok=True)

path_lista = base + "test_mapper_process.csv"
print(f"List for train = {path_lista}")
lista = pd.read_csv(path_lista)


# In[15]:


for _, row in tqdm(lista.iterrows(), total=len(lista), desc="Processing files", unit="file"):

    filewiki = row.get('filename')
    basename = row.get('basename')

    resp = f'{output_dir}TEST/{filewiki}'
    if not(os.path.isfile(resp) and os.path.getsize(resp) > 0):
        aux = f"{input_path}TEST/{filewiki}"
        print(f"Reading from {aux}")
    
        df = pd.read_csv(aux, sep='\t')
        df2 = df.sort_values('Line')

    

        output_rebel = open(resp, 'w', encoding="utf8")
        output_rebel.write(f"line_number\tmap\n")
    
        for line in df2.iterrows():
            word = line[1][1]
            lin = line[1][0]
    
            word = str(word) if pd.notna(word) else ""
            frase = word.strip().split("\t")
            test_sentence = ' '.join(map(str, frase))
    
            # Extração REBEL
            extracted_text = triplet_extractor.tokenizer.batch_decode(
                [triplet_extractor(test_sentence, return_tensors=True, return_text=False)[0]["generated_token_ids"]]
            )[0]
    
            # Traduzir apenas o conteúdo das tags
            def traduzir_tripletos(texto):
                def traduzir_grupo(match):
                    partes = match.group(0)
    
                    # Extrai os elementos
                    subj = re.search(r"<subj>(.*?)<obj>", partes)
                    obj = re.search(r"<obj>(.*?)($|<triplet>)", partes)
                    triplet = re.search(r"<triplet>(.*?)<subj>", partes)
    
                    s = subj.group(1).strip() if subj else ""
                    o = obj.group(1).strip() if obj else ""
                    t = triplet.group(1).strip() if triplet else ""
    
                    # Traduz individualmente
                    try:
                        t_pt = argos_translate(t, "en", "pt") if t else ""
                        s_pt = argos_translate(s, "en", "pt") if s else ""
                        o_pt = argos_translate(o, "en", "pt") if o else ""
                    except Exception as e:
                        print(f"[Falha tradução linha {lin}]: {e}")
                        return partes  # fallback: retorna como está
    
                    return f"<triplet> {t_pt} <subj> {s_pt} <obj> {o_pt}"
    
                # Traduz todos os grupos
                novo_texto = re.sub(r"<triplet>.*?<subj>.*?<obj>.*?(?=<triplet>|</s>)", traduzir_grupo, texto)
                return novo_texto
    
            try:
                texto_pt_formatado = traduzir_tripletos(extracted_text)
                output_rebel.write(f"{lin}\t{texto_pt_formatado}\n")
            except Exception as e:
                print(f"[Erro geral linha {lin}]: {e}")
                output_rebel.write(f"{lin}\t{extracted_text}\n")  # fallback
    
        output_rebel.close()
    else:
        print("Processing results already exist, skipping")


# # DEV

# In[16]:


os.makedirs(output_dir+"DEV", exist_ok=True)

path_lista = base + "validate_mapper_process.csv"
print(f"List for train = {path_lista}")
lista = pd.read_csv(path_lista)


# In[18]:


for _, row in tqdm(lista.iterrows(), total=len(lista), desc="Processing files", unit="file"):

    filewiki = row.get('filename')
    basename = row.get('basename')
    
    resp = f'{output_dir}DEV/{filewiki}'
    
    if not (os.path.isfile(resp) and os.path.getsize(resp) > 0):

        aux = f"{input_path}DEV/{filewiki}"
        print(f"Reading from {aux}")
    
        df = pd.read_csv(aux, sep='\t')
        df2 = df.sort_values('Line')




        output_rebel = open(resp, 'w', encoding="utf8")
        output_rebel.write(f"line_number\tmap\n")
    
        for line in df2.iterrows():
            word = line[1][1]
            lin = line[1][0]
    
            word = str(word) if pd.notna(word) else ""
            frase = word.strip().split("\t")
            test_sentence = ' '.join(map(str, frase))
    
            # Extração REBEL
            extracted_text = triplet_extractor.tokenizer.batch_decode(
                [triplet_extractor(test_sentence, return_tensors=True, return_text=False)[0]["generated_token_ids"]]
            )[0]
    
            # Traduzir apenas o conteúdo das tags
            def traduzir_tripletos(texto):
                def traduzir_grupo(match):
                    partes = match.group(0)
    
                    # Extrai os elementos
                    subj = re.search(r"<subj>(.*?)<obj>", partes)
                    obj = re.search(r"<obj>(.*?)($|<triplet>)", partes)
                    triplet = re.search(r"<triplet>(.*?)<subj>", partes)
    
                    s = subj.group(1).strip() if subj else ""
                    o = obj.group(1).strip() if obj else ""
                    t = triplet.group(1).strip() if triplet else ""
    
                    # Traduz individualmente
                    try:
                        t_pt = argos_translate(t, "en", "pt") if t else ""
                        s_pt = argos_translate(s, "en", "pt") if s else ""
                        o_pt = argos_translate(o, "en", "pt") if o else ""
                    except Exception as e:
                        print(f"[Falha tradução linha {lin}]: {e}")
                        return partes  # fallback: retorna como está
    
                    return f"<triplet> {t_pt} <subj> {s_pt} <obj> {o_pt}"
    
                # Traduz todos os grupos
                novo_texto = re.sub(r"<triplet>.*?<subj>.*?<obj>.*?(?=<triplet>|</s>)", traduzir_grupo, texto)
                return novo_texto
    
            try:
                texto_pt_formatado = traduzir_tripletos(extracted_text)
                output_rebel.write(f"{lin}\t{texto_pt_formatado}\n")
            except Exception as e:
                print(f"[Erro geral linha {lin}]: {e}")
                output_rebel.write(f"{lin}\t{extracted_text}\n")  # fallback
    
        output_rebel.close()
    else:
        print("Processing results already exist, skipping")


# In[ ]:




