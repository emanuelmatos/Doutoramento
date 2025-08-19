#!/usr/bin/env python
# coding: utf-8

# # Aplica MApper ao que está marcado como ENTITY

# ### INPUT = ..ateixeira/pipeline/ENTITY_DETECTOR_BY_BERT/OUTPUT_RESULTS/output_bert_final_DATASET.csv
# 
# ### OUTPUT = OUTPUT_RESULTS/results_mapper.txt

# # FAZ tudo

# In[1]:


import codecs
import sys
import pandas as pd
from SPARQLWrapper import SPARQLWrapper, JSON
import urllib.error
import time
from tqdm import tqdm
import os


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


base = pipeline
sys.path.append(base)

#-----------------------------------------------------------------------------------------
from config.configsite import * #basedir,inputdir
 
# Directory paths
#direxit = basedir
#filesdir = inputdir

inputdir=pipeline+"ENTITY_DETECTOR_BY_BERT/"+resultsdir
print(f"input dir = {inputdir}") # new AT april 2025


outputdir=pipeline+"MAPPER_ENTITY2CLASSES/"+resultsdir
print(f"output dir = {outputdir}") # new AT april 2025


# In[4]:


path_data = inputdir
#file_data = 'output_bert_text_city.csv'
file_data = f"output_bert_final_{dataset}.csv"


# In[ ]:


# Columns to read
cols_to_use = ['Word', 'LINE', 'BIO','POSITION','FILE']

# Read only those columns
df = pd.read_csv(path_data + file_data, sep='\t', usecols=cols_to_use)
#df0.rename(columns={'LINE':'Line',"BIO":"NER"},inplace = True)


# Step 2: Filter rows where NER is not equal to "O"
#filtered_df = df[df["NER"] != "O"].drop_duplicates().reset_index()

df0 = df
df0.rename(columns={'LINE':'Line',"BIO":"NER"},inplace = True)
filtered_df = df0[['FILE','Word','NER','Line','POSITION']].copy()
#filtered_df = df1.drop_duplicates().reset_index()
filtered_df

#### Saída no Mapper
output_folder = outputdir
output_file='results_mapper_v2.csv'

output_file2='results_mapper_by_word.csv'


# In[ ]:


# Define function to query Wikidata

#  INPUTS: endpoint + query
def get_results(endpoint_url, query):
    user_agent = f"WDQS-example Python/{sys.version_info[0]}.{sys.version_info[1]}"
    sparql = SPARQLWrapper(endpoint_url, agent=user_agent)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    try:
        return sparql.query().convert()
    except urllib.error.HTTPError:
        return None  # Suppress error messages
    except Exception:
        return None  # Suppress error messages


# In[9]:


# ---------------------------------------------------------------------------------------------
debug = True
debug2 = False
fast = True

# Define the SPARQL endpoint
endpoint_url = "https://query.wikidata.org/sparql"

#basem = "/home/ematos/phd/MAPPER/OUTPUT_RESULTS_WIKIMAPPER/"
#os.chdir(basem)

# Define output file
output_file_path = f"{output_folder}{output_file}"  # Update this path as needed
output_file_path2 = f"{output_folder}{output_file2}"  # Update this path as needed

entidade = ""
inside = False

# criar ficheiro e adicionar Header
output_wiki2 = open(output_file_path2, 'w', encoding="utf8") # word by word
output_wiki2.write("id_file\tLine\tPOSITION\tWORD\tNER\tMAPPER\n")  # Header


with open(output_file_path, 'w', encoding="utf8") as output_wiki:
    output_wiki.write("id_file\tLine\tPOSITION\tWORD\tNER\tresult\n")  # Header

    current_line = None
    n_word = 0
    
    # para todas as linhas 
    for index, row in tqdm(filtered_df.iterrows(), total=len(filtered_df), desc="Processing Words"):
        # informação na linha
        word = row["Word"]
        ner = row["NER"]
        Line = row["Line"]
        nword = row["POSITION"]
        id_file = os.path.splitext(row["FILE"])[0]

        #print(f"LINHA = {Line}\t{word}\t{ner}")

        # se ner começar por B_
        if ner.startswith("B-"):
            #print("Inicio de Entidade detetado")
            inside = True
            entidade = word
            ner_inicial = ner[2:]
            
        if ner.startswith("I-"):
            #print("continuação")
            inside = True
            entidade += " "+word


            
        if ner.startswith("O") and inside:
            #print("acabou")
            inside = False
            #print(f"Terminou: {entidade}")

            if debug2: 
                print(f"Vou fazer query com {entidade}")
            
            query = f"""
            SELECT ?typeLabel WHERE {{
              ?item rdfs:label "{entidade}"@en.
              ?item wdt:P31 ?type.
              SERVICE wikibase:label {{ bd:serviceParam wikibase:language "pt". }}
            }} LIMIT 1
            """
                   

     
        
            #else:
        #    query = f"""
        #    SELECT ?typeLabel WHERE {{
        #      ?item rdfs:label "{word}"@en.
        #      ?item wdt:P31 ?type.
        #      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "pt". }}
        #   }} LIMIT 1
        #    """

            if fast:
                time.sleep(0.01)
            else:
                time.sleep(0.1)

            results = get_results(endpoint_url, query)
            #print(f"Resultado = {results}")

            if results and "results" in results and "bindings" in results["results"]:
                bindings = results["results"]["bindings"]
                if bindings:
                    type_label = bindings[0].get('typeLabel', {}).get('value', 'O')
                else:
                    type_label = "O"
            else:
                type_label = "O"

        
        #if ner == "O":
        #    type_label = "NA"

            type_label = type_label.replace(" ","_").upper()
        
            if debug2:
                print(f"{entidade}\t{ner_inicial}\t{type_label}\n")
        
            output_wiki.write(f"{id_file}\t{Line}\t{nword}\t{entidade}\t{ner_inicial}\t{type_label}\n")
 

            # --- fazer output palavra a palavra
            palavra_na_entidade = entidade.split(" ")
            label1 = "B-"+ner_inicial
            label2 = "B-"+type_label
            #print(f"{palavra_na_entidade[0]}\t{label1}\t{label2}\n")
            output_wiki2.write(f"{id_file}\t{Line}\t{nword}\t{palavra_na_entidade[0]}\t{label1}\t{label2}\n")
    
            for palavra in palavra_na_entidade[1:]:
                label1 = "I-"+ner_inicial
                label2 = "I-"+type_label
                #print(f"{palavra}\t{label1}\t{label2}\n")
                output_wiki2.write(f"{id_file}\t{Line}\t{nword}\t{palavra}\t{label1}\t{label2}\n")
        

        if ner.startswith("O") and ~inside:
            output_wiki.write(f"{id_file}\t{Line}\t{nword}\t{word}\t{ner}\tO\n")
            output_wiki2.write(f"{id_file}\t{Line}\t{nword}\t{word}\t{ner}\tO\n")

output_wiki.close()
output_wiki2.close()

            


# In[ ]:




