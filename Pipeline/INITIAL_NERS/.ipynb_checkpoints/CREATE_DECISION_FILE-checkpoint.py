#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NER Data Processing Script
Created for robust and efficient handling of NER outputs with alignment and consensus building.
"""

import os
import pandas as pd
import sys
from tqdm import tqdm
from collections import Counter


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


#base = "/home/ematos/phd/pt/INITIAL_NERs/"
sys.path.append(pipeline)

# Configure directories
try:
    from config.configsite import * #filesdir, linguakitdir, inputdir, outputdir, basedir
except ImportError:
    print("Error: 'configsite' module not found. Ensure it is correctly configured.")
    sys.exit(1)
filesdir = f"{pipeline}/{dataset}_PROCESSED/"
lista_path = filesdir + "files_list.csv"
print(f"Using files list = {lista_path}")

lista = pd.read_csv(lista_path)
print(lista.head())

outputdir=pipeline+"INITIAL_NERS/"+resultsdir+"/"
print(f"output dir = {outputdir}") # new AT april 2025


# In[4]:


def wta(tag_str, threshold=2):
    """
    Determines the most common tag based on a threshold.
    """
    tags = tag_str.split(";")
    cnt = Counter(tags)
    most_common, count = cnt.most_common(1)[0]

    # Return "O" if no tag meets the threshold
    if count < threshold:
        return "O"
    return f"B-{most_common}" if most_common != "O" else "O"


# In[7]:


#####################################################################
debug = True
# Iterate through each file in the list
for _, row in tqdm(lista.iterrows(), total=len(lista), desc="Processing files", unit="file"):
    filename = row.get("filename")
    basename = row.get("basename")

    # Load the input DataFrame
    filename = "output.join"
    df = pd.read_csv(f"{outputdir}{basename}/{str(filename)}", sep=",")
    if debug:
        auxx = f"{outputdir}{basename}/{str(filename)}"
        print(f"Reading from {auxx}")

    # Initialize a new DataFrame for additional processing
    df2 = pd.DataFrame({
        "TAGS": ["O"] * len(df),        # Aggregated tags
        "TYPES": ["0"] * len(df),       # Entity types
        "NTAGS": [0] * len(df),         # Number of non-"O" tags
        "ENTITY": ["O"] * len(df),      # Entity labels (BIO format)
        "WTA": ["O"] * len(df),         # Winner Takes All label
        "CONSENSUS": ["O"] * len(df)    # Final consensus labels
    })

    # Map Linguakit entity codes to meaningful labels
    entity_mapping = {
        "NP00G00": "LOC",
        "NP00SP0": "PER",
        "NP00V00": "MISC",
        "NP00O00": "ORG"
    }
    # Replace entity codes in BIO2 column
    df["BIO2"] = df["BIO2"].replace(entity_mapping, regex=True)

    # Standardize BIO formats in BIO2 and BIO1 columns
    df["BIO2"] = df["BIO2"].replace({"U-": "B-"}, regex=True)
    df["BIO1"] = df["BIO1"].replace({"U-": "B-"}, regex=True)

    # Combine tags from BIO1, BIO2, and optionally BIO3
    df2["TAGS"] = df["BIO1"] + ";" + df["BIO2"] + ";" + df.get("BIO3", "O") + ";"

    # Extract types by removing BIO prefixes
    df2["TYPES"] = df2["TAGS"].replace({"[UBIL]-": ""}, regex=True)

    # Calculate NTAGS (number of non-"O" tags)
    df2["NTAGS"] = df2["TAGS"].apply(lambda x: 3 - x.count("O;"))

    # Assign ENTITY = "ENT" if BIO1, BIO2, or BIO3 is not "O"
    df["ENTITY"] = df.apply(
        lambda row: "ENT" if row["BIO1"] != "O" or row["BIO2"] != "O" or row["BIO3"] != "O" else "O",
        axis=1
    )

    # Apply WTA logic to TYPES column
    df2["WTA"] = df2["TYPES"].apply(lambda x: wta(x, threshold=2))

    # Merge results into the original DataFrame
    df["CONSENSUS"] = df2["WTA"]
    df["WTA"] = df2["WTA"]

    # Define the correct headers
    headers = [
        "WORD", "BIO1", "BIO2", "FILE", "LINE", "POSITION",
        "BIO3", "CONSENSUS", "WTA", "ENTITY"
    ]

    # Ensure all columns are present and properly ordered
    df = df.reindex(columns=headers)

    # Save the processed file with headers
    output_path = os.path.join(outputdir, str(basename), "output.decision")
    df.to_csv(output_path, sep="\t", index=False, header=True)

    # Save in CoNLL format (without headers)
    conll_path = output_path.replace(".decision", ".conll")
    df.to_csv(conll_path, sep=" ", index=False, header=False)

    # Append the results to the global decision.final file
    final_output_path = os.path.join(outputdir, "decision_final.csv")

    # Write header only if the file does not already exist
    if not os.path.exists(final_output_path):
        df.to_csv(final_output_path, sep="\t", index=False, header=True, mode="w")
    else:
        df.to_csv(final_output_path, sep="\t", index=False, header=False, mode="a")

print(f"Processing complete.\n Results saved in each folder and accumulated in '{final_output_path}'.")


# In[ ]:





# In[ ]:




