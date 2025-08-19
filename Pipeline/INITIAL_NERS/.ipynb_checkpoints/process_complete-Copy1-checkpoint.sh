#!/bin/bash

# Directory containing the files
SOURCE_DIR="/home/ematos/phd/NER_COMPLETE_SYSTEM_V1/INPUT_DATASET/"
DEST_DIR="/home/ematos/phd/NER_COMPLETE_SYSTEM_V1/OUTPUT_RESULTS/"

# Loop through all .txt files in the source directory
for file in "$SOURCE_DIR"/*.txt; do
  # Get the base filename without the directory
  filename=$(basename -- "$file")
  # Get the filename without the extension
  name="${filename%.*}"
  
  echo "$name"
  echo "$filename"
  
  # Copy the file to the current directory
  cp "$file" "$filename"
  
  # Process the file using the process_phase1 command
  ./process_phase.sh "$name"
  
  # Clean up output.* and input.txt files
  rm -f output.*
  rm -f input.txt
done

# Copy files to the NER_COMPLETE_SYSTEM directory
cp *.txt "$DEST_DIR"