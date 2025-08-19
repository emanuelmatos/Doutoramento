#!/bin/bash

# Enable strict error handling
set -euo pipefail
IFS=$'\n\t'

# Directories
SOURCE_DIR="/Volumes/hd_ext/phd/NER_COMPLETE_SYSTEM_V1/INPUT_DATASET/"
rm -f process.csv
CSV_FILE="process.csv"

# Step 1: Generate process.csv with all .txt files in the source directory
echo "Generating process.csv..."
echo "Filename,BaseName" > "$CSV_FILE"
for file in "$SOURCE_DIR"/*.txt; do
  if [[ -f "$file" ]]; then
    # Extract filename and base name
    filename=$(basename -- "$file")
    name="${filename%.*}"
    echo "$filename,$name" >> "$CSV_FILE"
  fi
done
echo "process.csv created."