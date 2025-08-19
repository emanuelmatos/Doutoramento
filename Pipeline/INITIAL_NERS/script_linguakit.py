#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Updated Linguakit NER processor with LINE and NWORD tracking per word.

Author: easm
"""

import os
import sys
import pandas as pd
import numpy as np
import re
from tqdm import tqdm


# In[2]:


#!ls


# In[3]:


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


# In[4]:


# ========================
# Diretórios e configurações
# ========================
base = pipeline
sys.path.append(base)

from config.configsite import *

#print(f"filesdir = {filesdir}") # new AT april 2025
print(f"pipeline = {pipeline}") # new AT april 2025

linguakitdir=pipeline+"INITIAL_NERS/Linguakit/"

outputdir=pipeline+"INITIAL_NERS/"+resultsdir
print(f"output dir = {outputdir}") # new AT april 2025

# Nome da lista de arquivos a processar
nome_lista = f'{pipeline}/{dataset}_PROCESSED/files_list.csv'


lista_path = os.path.join(pipeline, nome_lista)
print(lista_path)


# In[5]:


# Tenta carregar a lista
try:
    lista = pd.read_csv(lista_path)
except Exception as e:
    print(f"Erro ao ler o arquivo CSV: {e}")
    sys.exit(1)


# In[6]:


# ========================
# Função para garantir diretório
# ========================
def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
            print(f"Diretório criado: {directory}")
        except Exception as e:
            print(f"Erro ao criar diretório {directory}: {e}")
            sys.exit(1)

# ========================
# Processamento de arquivos
# ========================
for _, row in tqdm(lista.iterrows(), total=len(lista), desc="Processing files"):
    filename = row['filename']
    basename = row['basename']

    input_file = os.path.join(inputdir, str(filename)) # str to guarantee it works with nasmes with only integers
    output_dir = os.path.join(outputdir, str(basename))
    ensure_directory_exists(output_dir)

    outfilelinguakit = os.path.join(output_dir, "linguakit.txt")

    # CHECK IF EXISTS
    if os.path.exists(outfilelinguakit):
        print(f"WARNING! [LINGUAKIT NER] File with processing results already exists ({outfilelinguakit}).")
        continue

    # Executa o Linguakit
    os.chdir(linguakitdir)
    bash_command = f"linguakit tagger pt {input_file} -nec > {outfilelinguakit}"
    os.system(bash_command)

    # Leitura do texto original
    with open(input_file, 'r', encoding='utf-8') as f:
        original_text = f.read()

    # Divide texto em sentenças
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', original_text)

    # Leitura da saída do Linguakit
    try:
        data = pd.read_csv(outfilelinguakit, sep=" ", header=None, names=['words', 'tipo', 'nec'])
    except Exception as e:
        print(f"Erro ao ler output do Linguakit: {e}")
        continue

    # Limpeza e normalização
    data.replace({'_de_a_': '_da_', '_de_o_': '_do_', '_de_os_': '_dos_', '_de_as_': '_das_', '-': '_'}, regex=True, inplace=True)
    data.dropna(inplace=True)

    # Criação da estrutura de saída básica (sem LINE/NWORD ainda)
    output_rows = []
    file_parameter = filename

    for _, data_row in data.iterrows():
        palavra = data_row['words']
        nec = data_row['nec']

        if "_" in palavra:
            parts = palavra.split("_")
            for idx, part in enumerate(parts):
                if nec.startswith("NP0"):
                    tag = "B-" + nec if idx == 0 else "L-" + nec if idx == len(parts) - 1 else "I-" + nec
                else:
                    tag = "O"
                output_rows.append({
                    "WORD": part,
                    "BIO-LINGUAKIT": tag,
                    "FILE": file_parameter
                })
        else:
            tag = "U-" + nec if nec.startswith("NP0") else "O"
            output_rows.append({
                "WORD": palavra,
                "BIO-LINGUAKIT": tag,
                "FILE": file_parameter
            })

    output_df = pd.DataFrame(output_rows)

    # ========================
    # Atribui LINE e NWORD corretamente com base no texto
    # ========================
    final_rows = []
    word_pointer = 0

    for line_number, sentence in enumerate(sentences, start=1):
        words = sentence.strip().split()
        for nword, word in enumerate(words, start=1):
            if word_pointer >= len(output_df):
                break  # Proteção

            final_rows.append({
                "WORD": output_df.iloc[word_pointer]["WORD"],
                "BIO-LINGUAKIT": output_df.iloc[word_pointer]["BIO-LINGUAKIT"],
                "FILE": output_df.iloc[word_pointer]["FILE"]
            })
            word_pointer += 1

    # Se ainda houver palavras restantes
    for i in range(word_pointer, len(output_df)):
        final_rows.append({
            "WORD": output_df.iloc[i]["WORD"],
            "BIO-LINGUAKIT": output_df.iloc[i]["BIO-LINGUAKIT"],
            "FILE": output_df.iloc[i]["FILE"]
        })

    # DataFrame final ordenado
    output_df = pd.DataFrame(final_rows)

    # Salva resultado
    output_path = os.path.join(output_dir, 'output.linguakit')
    output_df.to_csv(output_path, index=False)
    print(f"Resultado salvo em: {output_path}")


# In[ ]:





# In[ ]:





# In[ ]:




