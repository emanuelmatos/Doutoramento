#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AllenNLP NER Pipeline

This script:
1. Downloads the official AllenNLP NER model (.tar.gz)
2. Loads it directly using Predictor
3. Processes sentences from text files listed in a CSV
4. Outputs token-level BIO tags in CSV format

Author: easm
"""


# In[2]:


import os
import sys
import requests
import pandas as pd
import nltk
from tqdm import tqdm
from allennlp.predictors.predictor import Predictor
import allennlp_models.tagging


# In[3]:


#!ls


# In[4]:


#from configsite import *
#print(f"filesdir = {filesdir}\n, inputdir = {inputdir}\n, outputdir={outputdir}\n")


# ### get basedir of pipeline

# In[5]:


import os

working_dir = os.getcwd()
print(working_dir)

parts = working_dir.split("/")
if parts[-1].startswith("pipeline"): ### BUG SOLVED MAY 2025
    pipeline = working_dir
else:
    pipeline ="/".join(parts[:-1])
print(pipeline)


# ### import variables from configsite.py, located at $base/config

# In[6]:


# --- Config ---

sys.path.append(os.path.abspath(pipeline))


# Project paths (from configsite.py)
try:
    from config.configsite import *  #  TODO: evitar o asterisco -- inputdir,dataset 
    #filesdir, inputdir, outputdir
except ImportError:
    print("[ERROR] Missing info in 'configsite.py'")
    sys.exit(1)

#print(f"filesdir = {filesdir}\n, inputdir = {inputdir}\n, outputdir={outputdir}\n")


# In[15]:


debug = False
BASE = pipeline
NER_MODEL_URL = "https://storage.googleapis.com/allennlp-public-models/ner-model-2020.02.10.tar.gz"
MODEL_DIR = "models"
MODEL_TARBALL = os.path.join(MODEL_DIR, "ner-model-2020.02.10.tar.gz")
OUTPUT_FILENAME = "output.allen.csv"
INPUT_CSV = f"{BASE}/{dataset}_PROCESSED/files_list.csv"  # AT Abril 2025

# TODO: mover para configsite.py

filesdir = f"{BASE}/{dataset}"

outputdir=pipeline+"INITIAL_NERS/"+resultsdir+"/"
print(f"output dir = {outputdir}") # new AT april 2025

lista_path = os.path.join(filesdir, INPUT_CSV)
os.makedirs(MODEL_DIR, exist_ok=True)


# In[16]:


# --- Model Downloader ---
def download_model():
    """Download the AllenNLP model tarball if missing."""
    if os.path.exists(MODEL_TARBALL):
        print(f"[INFO] Model already exists: {MODEL_TARBALL}")
        return
    print(f"[INFO] Downloading model from {NER_MODEL_URL}...")
    response = requests.get(NER_MODEL_URL, stream=True)
    if response.status_code != 200:
        raise RuntimeError(f"Failed to download model: HTTP {response.status_code}")
    with open(MODEL_TARBALL, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024):
            f.write(chunk)
    print("[INFO] Download complete.")

# --- Model Loader ---
def load_model():
    """Load the AllenNLP model directly from the .tar.gz archive."""
    try:
        print(f"[INFO] Downloading required NLTK resources...")
        nltk.download("punkt")
        print(f"[INFO] Loading AllenNLP model from: {MODEL_TARBALL}")
        predictor = Predictor.from_path(MODEL_TARBALL)
        print("[INFO] AllenNLP model loaded successfully.")
        return predictor
    except Exception as e:
        raise RuntimeError(f"Failed to load AllenNLP model: {e}")


# In[19]:


# --- File Processor ---
def process_file(filename, basename, predictor):
    """Run NER on each sentence from the file and write results."""
    input_path = os.path.join(inputdir, filename)
    print (f"Reading from file {input_path}")
    print (f"output dir = {outputdir}")

    #print(type(outputdir))
    #print(type(basename))
    
    output_dir = os.path.join(outputdir, str(basename))
    print (f"Output to file {output_dir}")
    
    output_path = os.path.join(output_dir, OUTPUT_FILENAME)
    os.makedirs(output_dir, exist_ok=True)

    

    try:
        with open(input_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"[WARN] File not found: {input_path}")
        return

    if debug:
        print(f"Lines = {lines}")  # AT Abril 2025
    
    results = []

    for line_num, text in enumerate(lines, 1):
        
        text = text.strip()
        if not text:
            if debug:
                print(f"line = {line_num} !! no text")
            continue
        else:
            if debug:
                print(f"line = {line_num} : {text}")
            
        try:
            response = predictor.predict(sentence=text)
            words = response.get("words", [])
            tags = response.get("tags", [])
            df = pd.DataFrame({
                "FILE": filename,
                "LINE": line_num,
                "POSITION": range(1, len(words) + 1),
                "WORD": words,
                "BIO_ALLEN": tags,
            })
            results.append(df)
        except Exception as e:
            print(f"[ERROR] Failed on line {line_num}: {e}\nTEXT={text}")

    if results:
        final_df = pd.concat(results, ignore_index=True)
        final_df.to_csv(output_path, index=False)
        print(f"[OK] Output saved to: {output_path}")
    else:
        print(f"[WARN] No lines processed for: {filename}")


# In[ ]:


# --- Main Execution ---
def main():
    try:
        filelist = pd.read_csv(lista_path)
    except Exception as e:
        print(f"[ERROR] Failed to load list: {e}")
        sys.exit(1)

    try:
        download_model()
        predictor = load_model()
    except Exception as e:
        print(f"[ERROR] Model setup failed: {e}")
        sys.exit(1)

    for _, row in tqdm(filelist.iterrows(), total=len(filelist), desc="Processing files", unit="file"):
        filename = row.get("filename")
        basename = row.get("basename")
        
        if not filename or not basename:
            print("[WARN] Skipping row with missing 'filename' or 'basename'")
            continue
        process_file(filename, basename, predictor)

if __name__ == "__main__":
    main()


# # Allen is at
# /home/ematos/miniconda3/envs/allenlp_env/lib/python3.9/site-packages/allennlp
# 

# In[ ]:




