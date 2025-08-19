#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 09:49:30 2021
Revised to handle FILE/LINE/POSITION logic.

Author: easm
"""

import os
import sys
import pandas as pd
import numpy as np
import re
from configsite import filesdir, linguakitdir


#-----------------------------------------------------------------------------------------
from configsite import filesdir

direxit=filesdir


#______________________________________________________________________________________



# Ensure a parameter (filename) is passed
if len(sys.argv) < 2:
    print("Usage: python process.py <filename>")
    sys.exit("ERROR: Missing parameter")

# Retrieve the parameter (e.g., "Abrolhos")
file_parameter = sys.argv[1]  # This will be "Abrolhos" in the example
print(f"Processing parameter: {file_parameter}")

# File and directory setup
input_file = "input.txt"
output_filename = f"{file_parameter}.linguakit"
outfilelinguakit = "linguakit.txt"

# Change to files directory
os.chdir(filesdir)
if os.path.exists(output_filename):
    print(f"Linguakit NER *** ALERT! File with processing results already exists ({output_filename}). Delete it if you want to process and rerun.")
    sys.exit("DONE")

# Linguakit processing
print(f"Using Linguakit directory: {linguakitdir}")
os.chdir(linguakitdir)

bash_command = f"linguakit tagger pt {filesdir}{input_file} -nec > {filesdir}{outfilelinguakit}"
print(f"Running: {bash_command}")
os.system(bash_command)

# Process Linguakit output
os.chdir(filesdir)

# Read the input file and split into sentences
with open(input_file, 'r', encoding='utf-8') as f:
    text = f.read()

# Split text into sentences using regex
sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)

# Map words to sentence and position
word_position_map = {}
for sentence_number, sentence in enumerate(sentences, start=1):
    words = sentence.strip().split()
    for position_in_sentence, word in enumerate(words, start=1):
        word_position_map[word] = (sentence_number, position_in_sentence)

# Read Linguakit output
data = pd.read_csv(outfilelinguakit, sep=" ", header=None, names=['words', 'tipo', 'nec'])
data.replace({'_de_a_': '_da_', '_de_o_': '_do_', '_de_os_': '_dos_', '_de_as_': '_das_', '-': '_'}, regex=True, inplace=True)
data.dropna(inplace=True)

# Initialize DataFrame for output
output_df = pd.DataFrame(columns=['WORD', 'BIO-LINGUAKIT', 'FILE'])

# Process each row of Linguakit output
for _, row in data.iterrows():
    palavra = row['words']
    nec = row['nec']

    # Find sentence number and word position
    sentence_number, position_in_sentence = word_position_map.get(palavra.split("_")[0], ("Unknown", "Unknown"))

    if "_" in palavra:  # Multi-word tokens
        parts = palavra.split("_")
        if nec.startswith("NP0"):  # Named entity multi-word
            # Add B, I, and L tags for multi-word named entities
            for idx, part in enumerate(parts):
                tag = "B-" + nec if idx == 0 else "L-" + nec if idx == len(parts) - 1 else "I-" + nec
                output_df = pd.concat([output_df, pd.DataFrame([{
                    "WORD": part,
                    "BIO-LINGUAKIT": tag,
                    "FILE": file_parameter
                }])], ignore_index=True)
        else:  # Non-entity multi-word
            for part in parts:
                output_df = pd.concat([output_df, pd.DataFrame([{
                    "WORD": part,
                    "BIO-LINGUAKIT": "O",
                    "FILE": file_parameter
                }])], ignore_index=True)
    else:  # Single-word tokens
        tag = "U-" + nec if nec.startswith("NP0") else "O"
        output_df = pd.concat([output_df, pd.DataFrame([{
            "WORD": palavra,
            "BIO-LINGUAKIT": tag,
            "FILE": file_parameter
        }])], ignore_index=True)

# Save to output file
output_df.to_csv('output.linguakit', index=False)
#output_df.to_csv(output_filename, index=False)
output_df.to_csv(direxit+'output.linguakit', index=False)
#output_df.to_csv(direxit+output_filename, index=False)

print(f"Results saved to {output_filename}.")