#!/usr/bin/env python
# coding: utf-8

# # Aplica Mapper No Rebel

# In[1]:


import sys
import time
import pandas as pd
from tqdm import tqdm
import os
import glob
import re
 


# In[2]:


import pandas as pd
 
import urllib.error
from SPARQLWrapper import SPARQLWrapper, JSON


# In[3]:


# Função para consultar Wikidata
def get_results(endpoint_url, query):
    user_agent = f"WDQS-example Python/{sys.version_info[0]}.{sys.version_info[1]}"
    sparql = SPARQLWrapper(endpoint_url, agent=user_agent)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    try:
        return sparql.query().convert()
    except urllib.error.HTTPError:
        return None
    except Exception:
        return None


# In[4]:


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


# In[12]:


base = pipeline
sys.path.append(base)


from config.configsite import * #basedir,inputdir
 

inputdir=pipeline+"ENTITY_DETECTOR_BY_BERT/"+resultsdir
print(f"input dir = {inputdir}") # new AT april 2025

input_path = pipeline+'REBEL_ENTITY2CLASSES/'+resultsdir+"/RESULTS/"
input_dataset = input_path
print(f"input path = {input_path}")

input_dir = input_path

output_dir = pipeline+"REBEL_ENTITY2CLASSES/"+resultsdir+"CLASSES/"
print(f"output dir = {output_dir}")

os.makedirs(output_dir, exist_ok=True)

os.makedirs(f"{output_dir}TRAIN", exist_ok=True)


# # TRAIN

# In[13]:


# Caminho correto
#folder_path = '/home/ematos/phd/pt/REBEL_ENTITY2CLASSES/RESULTS/TRAIN/'
folder_path = input_path+"/TRAIN/"
print(f"Folder path = {folder_path}")

# Lista de arquivos .txt
file_paths = glob.glob(os.path.join(folder_path, "ou*.txt"))

# Função para extrair basename
def extract_basename(filename):
    match = re.match(r"output_wikidata_mapping_pt_(.+)\.txt", filename)
    return match.group(1) if match else None

# Corrigido: usar file_paths, não files
lista = pd.DataFrame({
    "filename": [os.path.basename(f) for f in file_paths],
    "basename": [extract_basename(os.path.basename(f)) for f in file_paths]
})

print(f"\n{len(lista)} files processed.") 
print(lista.tail())


# In[14]:


# SPARQL endpoint
endpoint_url = "https://query.wikidata.org/sparql"

# Loop sobre arquivos
for _, row in tqdm(lista.iterrows(), total=len(lista), desc="Processing files", unit="file"):

    filewiki = row.get('filename')
    basename = row.get('basename')

    aux = f"{input_path}TRAIN/{filewiki}"
    print(f"Processing {aux}")
    df = pd.read_csv(aux, sep='\t')

    resp = f"{output_dir}TRAIN/results_wiki_train_pt_{basename.lower()}.txt"
    print(f"Saving to {resp}")
    with open(resp, 'w', encoding="utf8") as output_dbmapper:
        output_dbmapper.write("line\tmapper\tbase\tresult\n")

        for _, data_row in df.iterrows():
            base = data_row.get("entity", "NA")
            line_num = data_row.get("line_number", "NA")
            mapper = data_row.get("wikidata_id", "NA")

            result_label = "NA"

            if isinstance(mapper, str) and mapper.startswith("Q"):
                query = f"""
                SELECT ?typeLabel WHERE {{
                  wd:{mapper} wdt:P31 ?type.
                  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "pt". }}
                }} LIMIT 1
                """
                time.sleep(1)  # Evita sobrecarregar o endpoint
                results = get_results(endpoint_url, query)
                
                try:
                    bindings = results.get("results", {}).get("bindings", [])
                    if bindings:
                        result_label = bindings[0]['typeLabel']['value']
                except Exception:
                    result_label = "ERRO"

            output_dbmapper.write(f"{line_num}\t{mapper}\t{base}\t{result_label}\n")
            #print(f"{base}, {line_num}, {mapper}, {result_label}")


# In[ ]:




