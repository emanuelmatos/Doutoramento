#!/bin/bash

# Enable strict error handling
set -euo pipefail
IFS=$'\n\t'

# Directories
SOURCE_DIR="/home/ematos/phd/NER_COMPLETE_SYSTEM_V1/INPUT_DATASET/"
DEST_DIR="/home/ematos/phd/NER_COMPLETE_SYSTEM_V1/OUTPUT_RESULTS/"
CSV_FILE="process4.csv"
PROCESSED_LIST="list_processed4.csv"

# Create or clear the processed list CSV file
if [[ ! -f "$PROCESSED_LIST" ]]; then
  echo "Filename,Base Name" > "$PROCESSED_LIST"
else
  echo "Appending to existing $PROCESSED_LIST"
fi

# Read process.csv and process each file using a for loop
echo "Reading process.csv and processing files..."
lines=($(tail -n +2 "$CSV_FILE")) # Skip the header line
for line in "${lines[@]}"; do
  IFS=',' read -r filename name <<< "$line"

  echo "Processing file: $filename with base name: $name ..........................................................................."

  # Ensure the output directory exists for this file
  output_dir="$DEST_DIR/$name"
  mkdir -p "$output_dir"
  
  # Copy the file to the current directory for processing
  cp "$SOURCE_DIR/$filename" "$filename"

  # Process the file using process_phase.sh
  python process_phase.py "$name"

  # Move all output files to the specific output directory
  mv output.* "$output_dir/" 2>/dev/null || true
  mv input.txt "$output_dir/" 2>/dev/null || true

  # Clean up temporary files
  rm -f output.*
  rm -f input.txt

  # Add the processed file and base name to the processed list
  echo "$filename,$name" >> "$PROCESSED_LIST"
done

echo "All files processed. Results saved to $DEST_DIR."
echo "Processed files logged in $PROCESSED_LIST."