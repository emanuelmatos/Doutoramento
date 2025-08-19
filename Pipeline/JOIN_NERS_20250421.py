#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NER Output Alignment Script
Created on Tue Mar 16 09:49:30 2021
Updated for clarity, optimization, and robustness.
"""

import os
import pandas as pd
import sys
from tqdm import tqdm


# In[2]:


import os

working_dir = os.getcwd()
print(working_dir)

parts = working_dir.split("/")
if parts[-1] == "pipeline":
    pipeline = working_dir
else:
    pipeline ="/".join(parts[:-1])
print(pipeline)


# In[3]:


#base = "/home/ematos/phd/pt/INITIAL_NERs/"
sys.path.append(pipeline)

# Configure directories
try:
    from config.configsite import * #filesdir, linguakitdir, inputdir, outputdir, basedir
except ImportError:
    print("Error: 'configsite' module not found. Ensure it is correctly configured.")
    sys.exit(1)

filesdir = f"{pipeline}/{dataset}_PROCESSED/"
lista_path = filesdir + "/files_list.csv"

lista = pd.read_csv(lista_path)
lista.head()

outputdir=pipeline+"INITIAL_NERS/"+resultsdir+"/"
print(f"output dir = {outputdir}") # new AT april 2025


# In[6]:


def log_debug(message):
    """Writes debug messages to the debug file if debugging is enabled."""
    if DEBUG:
        debug_file.write(message + "\n")


# In[7]:


def align_words(data1, data2, bio_col1, bio_col2, file_col, line_col, pos_col, window=6):
        """
        Aligns words between two NER datasets and returns a merged DataFrame.

        Args:
            data1 (pd.DataFrame): First dataset.
            data2 (pd.DataFrame): Second dataset.
            bio_col1 (str): Column name for BIO tags in the first dataset.
            bio_col2 (str): Column name for BIO tags in the second dataset.
            file_col (str): Column name for the file identifier.
            line_col (str): Column name for the line identifier.
            pos_col (str): Column name for the position identifier.
            window (int): Sliding window size for handling mismatches.

        Returns:
            pd.DataFrame: Aligned DataFrame.
        """
        i, j = 0, 0
        len1, len2 = len(data1), len(data2)
        merged_data = []

        while i < len1:
            word1 = data1.iloc[i]["WORD"]
            word2 = data2.iloc[j]["WORD"] if j < len2 else None
            log_debug(f"** {i}: word1={word1}  {j}: word2={word2}")

            if word1 == word2:
                merged_data.append({
                    "WORD": word1,
                    bio_col1: data1.iloc[i][bio_col1],
                    bio_col2: data2.iloc[j][bio_col2] if j < len2 else "O",
                    "FILE": data1.iloc[i][file_col],
                    "LINE": data1.iloc[i][line_col],
                    "POSITION": data1.iloc[i][pos_col]
                })
                i += 1
                j += 1
            else:
                # Handle misalignment with a sliding window
                next_words1 = data1.iloc[i:i + window]["WORD"].tolist()
                next_words2 = data2.iloc[j:j + window]["WORD"].tolist()

                if word1 in next_words2:
                    j += next_words2.index(word1)
                elif word2 in next_words1:
                    i += next_words1.index(word2)
                else:
                    merged_data.append({
                        "WORD": word1,
                        bio_col1: data1.iloc[i][bio_col1],
                        bio_col2: "O",
                        "FILE": data1.iloc[i][file_col],
                        "LINE": data1.iloc[i][line_col],
                        "POSITION": data1.iloc[i][pos_col]
                    })
                    i += 1

        return pd.DataFrame(merged_data)


# In[11]:


# Iterate through each file in the list
debug = True
DEBUG = False

for _, row in tqdm(lista.iterrows(), total=len(lista), desc="Processing files", unit="file"):
    filename = row.get("filename")
    basename = row.get("basename")

    print(f"\nProcessing {filename}")
 

    # Paths and filenames
    #FILENAMES = ["output.allen.csv", "output.linguakit", "output.dbpediaNER"]
    FILENAMES = ["output.allen", "output.linguakit", "output.dbpediaNER"] # novo 20 june

    # Load input files
    try:
        filename_allen = f"{outputdir}{basename}/{FILENAMES[0]}"
        if debug:
            print(f"file Allen = {filename_allen}")
            
        data_allen = pd.read_csv(filename_allen, sep=",")
        data_allen.rename(columns={"BIO_ALLEN": "BIO1"}, inplace=True)
    except Exception as e:
        print(f"\n **** Error = {e}\nCONTINUING")
        continue
    

    try:
        filename_linguakit= f"{outputdir}{basename}/{FILENAMES[1]}"
        if debug:
            print(f"file Linguakit = {filename_linguakit}")
            
        data_linguakit = pd.read_csv(filename_linguakit, sep=",")
        data_linguakit.rename(columns={"BIO-LINGUAKIT": "BIO2"}, inplace=True)
        print(data_linguakit.head(3))
    except Exception as e:
        print(f"\n **** Error = {e}\nCONTINUING")
        continue

    try:
        auxx = f"{outputdir}{basename}/{FILENAMES[2]}"
        if debug:
            print(f"file dbpedia = {auxx}")
        data_dbpedia = pd.read_csv(auxx, sep=",")
        data_dbpedia.rename(columns={"BIO": "BIO3", "Token": "WORD"}, inplace=True)
 
    except Exception as e:
        auxx = f"{outputdir}{basename}/{FILENAMES[2]}"
        #data_dbpedia = pd.read_csv(f"{outputdir}{basename}/{FILENAMES[1]}", sep=",") # faked
        #print(f"\n****ERROR!  No DBPEDIA files {auxx}\n\n")
        #data_dbpedia.rename(columns={"BIO-LINGUAKIT": "BIO3"}, inplace=True)
        print(f"\n **** Error = {e}\nCONTINUING")
        continue


    # Initialize DataFrame for merged results
    df_result = pd.DataFrame(columns=["WORD", "BIO1", "BIO2", "BIO3", "FILE", "LINE", "POSITION"])

    # Debugging file setup
    if DEBUG:
        debug_file = open("debug-join.txt", "w", encoding="utf-8")



    

    # Ensure required columns exist
    required_columns = ["WORD", "BIO1", "FILE", "LINE", "POSITION"]
    for col in required_columns:
        if col not in data_allen.columns:
            raise ValueError(f"Missing required column '{col}' in data_allen.")

    # Merge AllenNLP and Linguakit results
    print("Merging AllenNLP and Linguakit results...")
    df_result = align_words(
        data_allen, data_linguakit, "BIO1", "BIO2", "FILE", "LINE", "POSITION"
    )

    # Merge DBpedia results
    print("Merging DBpedia results...")
    dbpedia_dict = dict(zip(data_dbpedia["WORD"], data_dbpedia["BIO3"]))

    df_result["BIO3"] = df_result["WORD"].map(lambda word: dbpedia_dict.get(word, "O"))

    # Clean BIO3 column
    df_result["BIO3"] = df_result["BIO3"].replace(
        {"[A-Za-z]+;": "", "U-": "B-", "L-": "I-"}, regex=True
    ).replace(
        {"Person$": "PER", "Place$": "LOC", "Organisation$": "ORG"}, regex=True
    )

    # Save the output
    output_file = f"{outputdir}{basename}/output.join"
    print(f"NER alignment completed. Output will be saved in '{output_file}'.")
    
    df_result.to_csv(output_file, index=False)
    if DEBUG:
        debug_file.close()

    #print(f"NER alignment completed. Output saved as '{output_file}'.")


# In[ ]:




