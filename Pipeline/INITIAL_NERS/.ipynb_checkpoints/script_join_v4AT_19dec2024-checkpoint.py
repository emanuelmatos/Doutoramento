#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NER Output Alignment Script
Created on Tue Mar 16 09:49:30 2021
Updated for clarity, optimization, and robustness.
"""

import os
import pandas as pd
from configsite import filesdir

# Debugging flag
DEBUG = True

# Paths and filenames
os.chdir(filesdir)
FILENAMES = ['output.allen', 'output.linguakit', 'output.dbpediaNER']

# Load input files
data_allen = pd.read_csv(FILENAMES[0], sep=",")
data_linguakit = pd.read_csv(FILENAMES[1], sep=",")
data_dbpedia = pd.read_csv(FILENAMES[2], sep=",")

# Initialize DataFrame for merged results
df_result = pd.DataFrame(columns=['WORD', 'BIO-ALLEN', 'BIO-LINGUAKIT', 'BIO-DBPEDIA'])

# Debugging file setup
if DEBUG:
    debug_file = open("debug-join.txt", "w", encoding="utf-8")


def log_debug(message):
    """Writes debug messages to the debug file if debugging is enabled."""
    if DEBUG:
        debug_file.write(message + "\n")


def align_words(data1, data2, bio_col1, bio_col2, window=6):
    """
    Aligns words between two NER datasets and returns a merged DataFrame.

    Args:
        data1 (pd.DataFrame): First dataset.
        data2 (pd.DataFrame): Second dataset.
        bio_col1 (str): Column name for BIO tags in the first dataset.
        bio_col2 (str): Column name for BIO tags in the second dataset.
        window (int): Sliding window size for handling mismatches.

    Returns:
        pd.DataFrame: Aligned DataFrame.
    """
    i, j = 0, 0
    len1, len2 = len(data1), len(data2)
    merged_data = []

    while i < len1:
        word1 = data1.iloc[i, 0]
        word2 = data2.iloc[j, 0] if j < len2 else None
        log_debug(f"** {i}: word1={word1}  {j}: word2={word2}")

        if word1 == word2:
            merged_data.append({
                "WORD": word1,
                bio_col1: data1.iloc[i, 1],
                bio_col2: data2.iloc[j, 1]
            })
            i += 1
            j += 1
        else:
            # Handle misalignment with a sliding window
            next_words1 = data1.iloc[i:i + window, 0].tolist()
            next_words2 = data2.iloc[j:j + window, 0].tolist()

            if word1 in next_words2:
                j += next_words2.index(word1)
            elif word2 in next_words1:
                i += next_words1.index(word2)
            else:
                merged_data.append({
                    "WORD": word1,
                    bio_col1: data1.iloc[i, 1],
                    bio_col2: "O"
                })
                i += 1

    return pd.DataFrame(merged_data)


# Merge AllenNLP and Linguakit results
print("Merging AllenNLP and Linguakit results...")
df_result = align_words(data_allen.iloc[:, :2], data_linguakit.iloc[:, 0:2], "BIO-ALLEN", "BIO-LINGUAKIT")

# Merge DBpedia results
print("Merging DBpedia results...")
dbpedia_dict = dict(zip(data_dbpedia.loc[:,'WORD'], data_dbpedia.loc[:,'BIO-DBPEDIA']))

df_result['BIO-DBPEDIA'] = df_result['WORD'].map(lambda word: dbpedia_dict.get(word, "O"))

# Clean BIO3 column
df_result["BIO-DBPEDIA"] = df_result["BIO-DBPEDIA"].replace(
    {"[A-Za-z]+;": "", "U-": "B-", "L-": "I-"}, regex=True
).replace(
    {"Person$": "PER", "Place$": "LOC", "Organisation$": "ORG"}, regex=True
)

# Save the output
df_result.rename(
    columns={"BIO-ALLEN": "BIO1", "BIO-LINGUAKIT": "BIO2", "BIO-DBPEDIA": "BIO3"},
    inplace=True
)
df_result.to_csv("output.join", index=False)
if DEBUG:
    debug_file.close()

print("NER alignment completed. Output saved as 'output.join'.")