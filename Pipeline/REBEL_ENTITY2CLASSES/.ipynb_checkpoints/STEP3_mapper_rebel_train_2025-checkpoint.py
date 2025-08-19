#!/usr/bin/env python
# coding: utf-8

# # TRAIN

# In[1]:


import sys
import time
import pandas as pd
from tqdm import tqdm
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


# In[4]:


sys.path.append(pipeline)

from config.configsite import * #basedir,inputdir

outputdir=pipeline+"REBEL_ENTITY2CLASSES/"+resultsdir
print(f"output dir = {outputdir}") # new AT april 2025
base = outputdir


# In[20]:


sys.path.append(pipeline)

from config.configsite import * #basedir,inputdir


# In[26]:


inputdir=pipeline+"REBEL_ENTITY2CLASSES/"+resultsdir+"RESULTS"
print(f"inputdir = {inputdir}") # new AT 


# In[ ]:





# In[8]:


#!pip install wikimapper


# In[17]:


#!pip install  deep_translator


# # WIKIMAPPER condiguration
# The needed mappings for PT and EN are at $pipeline/AUXILIARY/WIKIMAPPER
# 

# In[15]:


from wikimapper import WikiMapper
import re
import pandas as pd
from deep_translator import GoogleTranslator
from tqdm import tqdm  # Progress bar for multiple files

# Initialize WikiMapper for Portuguese and English
aux = pipeline+"AUXILIARY/WIKIMAPPER/index_ptwiki-latest.db"
print(f"PT Mapper = {aux}")
mapper_pt = WikiMapper(aux)
aux = pipeline+"AUXILIARY/WIKIMAPPER/index_enwiki-latest.db"
print(f"PT Mapper = {aux}")
mapper_en = WikiMapper(aux)


# # Setup Translator

# In[19]:


# Initialize translator (Portuguese to English)
translator = GoogleTranslator(source='pt', target='en')


# # Process TRAIN

# In[31]:


path_lista = base + "train_mapper_process.csv"
lista = pd.read_csv(path_lista)
lista.tail()


# In[30]:


for _, row in tqdm(lista.iterrows(), total=len(lista), desc="Processing files", unit="file"):
    filewiki = row.get('filename')
    basename = row.get('basename')

    result_file_name = f"{inputdir}/TRAIN/output_wikidata_mapping_pt_{basename}.txt"

    if (os.path.isfile(result_file_name) and os.path.getsize(result_file_name) > 0):
        print(f"! Results already exist, Skipping {inputdir}/TRAIN/{filewiki}")
        continue
    else:
    
        aux = f"{inputdir}/TRAIN/{filewiki}"
        print(f"Reading from {aux}")
        data = pd.read_csv(aux, sep = '\t')
    
        df = data
        #print(df)
        # Regular expression to extract entities from <triplet> tags
        triplet_pattern = re.compile(r"<triplet>\s*(.*?)\s*<subj>")
        
        # Open output files
        with open(result_file_name, "w", encoding="utf8") as output_file_pt:
             #open(f"{outpath}/EN/output_wikidata_mapping_en_{basename}.txt", "w", encoding="utf8") as output_file_en:
            
            output_file_pt.write("line_number\tentity\twikidata_id\n")
            #output_file_en.write("line_number\tentity\ttranslated_entity\twikidata_id\n")
            
            # Process each row in the DataFrame
            for index, row in df.iterrows():
                line_number = row["line_number"]
                content = row["map"]
                entities = triplet_pattern.findall(content)
        
                for entity in entities:
                    entity = entity.strip()
                    try:
                        translated_entity = translator.translate(entity)
                        mapped_entity = translated_entity.replace(" ", "_")
                    except Exception as e:
                        print(f"Skipping entity '{entity}' due to translation error: {e}")
                        continue
        
                    # Wikidata ID lookup for both PT and EN
                    wikidata_id_pt = mapper_pt.title_to_id(entity.replace(" ", "_"))
                    wikidata_id_en = mapper_en.title_to_id(mapped_entity)
                    
                    # Write Portuguese mapping
                    if wikidata_id_pt:
                        output_file_pt.write(f"{line_number}\t{entity}\t{wikidata_id_pt}\n")
                    else:
                        output_file_pt.write(f"{line_number}\t{entity}\tNot Found\n")
                
    print("Translation and Wikidata mapping completed. Results saved in 'output_wikidata_mapping_pt.txt'.")


# # TEST

# In[32]:


path_lista = base + "test_mapper_process.csv"
lista = pd.read_csv(path_lista)
lista.tail()


# In[34]:


for _, row in tqdm(lista.iterrows(), total=len(lista), desc="Processing files", unit="file"):
    filewiki = row.get('filename')
    basename = row.get('basename')

    result_file_name = f"{inputdir}/TEST/output_wikidata_mapping_pt_{basename}.txt"

    if (os.path.isfile(result_file_name) and os.path.getsize(result_file_name) > 0):
        print(f"! Results already exist, Skipping {inputdir}/TEST/{filewiki}")
        continue
    else:

        data = pd.read_csv(f"{inputdir}/TEST/{filewiki}", sep = '\t')
    
        df = data
        #print(df)
        # Regular expression to extract entities from <triplet> tags
        triplet_pattern = re.compile(r"<triplet>\s*(.*?)\s*<subj>")
        
        # Open output files
        with open(f"{inputdir}/TEST/output_wikidata_mapping_pt_{basename}.txt", "w", encoding="utf8") as output_file_pt:
             #open(f"{outpath}/EN/output_wikidata_mapping_en_{basename}.txt", "w", encoding="utf8") as output_file_en:
            
            output_file_pt.write("line_number\tentity\twikidata_id\n")
            #output_file_en.write("line_number\tentity\ttranslated_entity\twikidata_id\n")
            
            # Process each row in the DataFrame
            for index, row in df.iterrows():
                line_number = row["line_number"]
                content = row["map"]
                entities = triplet_pattern.findall(content)
        
                for entity in entities:
                    entity = entity.strip()
                    try:
                        translated_entity = translator.translate(entity)
                        mapped_entity = translated_entity.replace(" ", "_")
                    except Exception as e:
                        print(f"Skipping entity '{entity}' due to translation error: {e}")
                        continue
        
                    # Wikidata ID lookup for both PT and EN
                    wikidata_id_pt = mapper_pt.title_to_id(entity.replace(" ", "_"))
                    wikidata_id_en = mapper_en.title_to_id(mapped_entity)
                    
                    # Write Portuguese mapping
                    if wikidata_id_pt:
                        output_file_pt.write(f"{line_number}\t{entity}\t{wikidata_id_pt}\n")
                    else:
                        output_file_pt.write(f"{line_number}\t{entity}\tNot Found\n")
                
    print("Translation and Wikidata mapping completed. Results saved in 'output_wikidata_mapping_pt.txt'.")


# # DEV

# In[37]:


path_lista = base + "validate_mapper_process.csv"
lista = pd.read_csv(path_lista)
lista.tail()


# In[38]:


for _, row in tqdm(lista.iterrows(), total=len(lista), desc="Processing files", unit="file"):
    filewiki = row.get('filename')
    basename = row.get('basename')

    result_file_name = f"{inputdir}/DEV/output_wikidata_mapping_pt_{basename}.txt"

    if (os.path.isfile(result_file_name) and os.path.getsize(result_file_name) > 0):
        print(f"! Results already exist, Skipping {inputdir}/DEV/{filewiki}")
        continue
    else:

        data = pd.read_csv(f"{inputdir}/DEV/{filewiki}", sep = '\t')
    
        df = data
        #print(df)
        # Regular expression to extract entities from <triplet> tags
        triplet_pattern = re.compile(r"<triplet>\s*(.*?)\s*<subj>")
        
        # Open output files
        with open(f"{inputdir}/DEV/output_wikidata_mapping_pt_{basename}.txt", "w", encoding="utf8") as output_file_pt:
             #open(f"{outpath}/EN/output_wikidata_mapping_en_{basename}.txt", "w", encoding="utf8") as output_file_en:
            
            output_file_pt.write("line_number\tentity\twikidata_id\n")
            #output_file_en.write("line_number\tentity\ttranslated_entity\twikidata_id\n")
            
            # Process each row in the DataFrame
            for index, row in df.iterrows():
                line_number = row["line_number"]
                content = row["map"]
                entities = triplet_pattern.findall(content)
        
                for entity in entities:
                    entity = entity.strip()
                    try:
                        translated_entity = translator.translate(entity)
                        mapped_entity = translated_entity.replace(" ", "_")
                    except Exception as e:
                        print(f"Skipping entity '{entity}' due to translation error: {e}")
                        continue
        
                    # Wikidata ID lookup for both PT and EN
                    wikidata_id_pt = mapper_pt.title_to_id(entity.replace(" ", "_"))
                    wikidata_id_en = mapper_en.title_to_id(mapped_entity)
                    
                    # Write Portuguese mapping
                    if wikidata_id_pt:
                        output_file_pt.write(f"{line_number}\t{entity}\t{wikidata_id_pt}\n")
                    else:
                        output_file_pt.write(f"{line_number}\t{entity}\tNot Found\n")
                
    print("Translation and Wikidata mapping completed. Results saved in 'output_wikidata_mapping_pt.txt'.")


# In[ ]:




