#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 09:49:30 2021
Revised to include the position of each word within a sentence.

Author: easm
"""

import os
import sys
import pandas as pd
from allennlp.predictors.predictor import Predictor
import allennlp_models.tagging

from allennlp.data.token_indexers import TokenCharactersIndexer
# Set min_padding_length explicitly, usually equal to the max kernel size of the CnnEncoder
token_characters_indexer = TokenCharactersIndexer(min_padding_length=5)

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
print(f"Processing with parameter: {file_parameter}")

# Configurations
from configsite import dataDir

# Input and output file names
INPUT_FILE = "input.txt"
output_filename = f"{file_parameter}.allen"  # Output file name dynamically based on parameter

# Change working directory to the data directory
os.chdir(dataDir)

# Check if output file already exists
if os.path.exists(output_filename):
    print(f"ALLEN NER *** ALERT! File with processing results already exists ({output_filename}). "
          f"Delete it if you want to process and rerun.")
    sys.exit("DONE")

# Read input data
try:
    with open(INPUT_FILE, 'r', encoding='utf8') as file:
        input_lines = file.readlines()  # Read file line by line
except FileNotFoundError:
    print(f"Error: Input file '{INPUT_FILE}' not found in {dataDir}.")
    sys.exit("ERROR: File Not Found")

# Load AllenNLP NER model
try:
    predictor = Predictor.from_path(
        "https://storage.googleapis.com/allennlp-public-models/ner-model-2020.02.10.tar.gz"
    )
except Exception as e:
    print(f"Error loading AllenNLP model: {e}")
    sys.exit("ERROR: Model Load Failure")

# Process each line and collect results
processed_data = []

for line_num, line_text in enumerate(input_lines, start=1):
    if line_text.strip():  # Skip empty lines
        try:
            response = predictor.predict(line_text.strip())
            words = response["words"]
            tags = response["tags"]

            # Add word positions
            word_positions = list(range(1, len(words) + 1))

            # Create a DataFrame for this line
            ner_data = pd.DataFrame({
                "WORD": words,
                "BIO_ALLEN": tags,
                "FILE": file_parameter,
                "LINE": line_num,
                "POSITION": word_positions
            })

            processed_data.append(ner_data)
        except Exception as e:
            print(f"Error processing line {line_num}: {e}")

# Combine results into a single DataFrame
try:
    if processed_data:
        final_df = pd.concat(processed_data, ignore_index=True)
    else:
        raise ValueError("No data was processed.")
except Exception as e:
    print(f"Error combining DataFrame: {e}")
    sys.exit("ERROR: DataFrame Combination Failure")

# Save the processed data to the output file
try:
    #final_df.to_csv(output_filename, index=False)
    final_df.to_csv('output.allen', index=False)
    #final_df.to_csv(direxit+output_filename, index=False)
    final_df.to_csv(direxit+'output.allen', index=False)
    print(f"Processing complete. Results saved to {output_filename}.")
except Exception as e:
    print(f"Error saving output file: {e}")
    sys.exit("ERROR: File Save Failure")