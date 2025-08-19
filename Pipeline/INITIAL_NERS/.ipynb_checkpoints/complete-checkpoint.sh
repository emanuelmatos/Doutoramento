#!/bin/bash

# Directories for input and output
SOURCE_DIR="/home/ematos/phd/NER_COMPLETE_SYSTEM_V1/INPUT_DATASET/"
DEST_DIR="/home/ematos/phd/NER_COMPLETE_SYSTEM_V1/OUTPUT_RESULTS/"

# Function to process a single file
process_file() {
  local file="$1"
  local current="$2"
  local total="$3"
  local filename=$(basename -- "$file")
  local name="${filename%.*}"

  # Calculate percentage of completion
  local percent=$((current * 100 / total))

  echo "Processing file $current of $total ($percent%): $filename"

  # Copy the source file to the working directory
  cp "$file" "$filename"

  # Processing pipeline
  echo "Running preprocessing and NER pipeline for $filename..."

  # Preprocessing step
  cp "$filename" aux1.txt
  python3 preprocess.py

  # Named Entity Recognition (NER) steps
  python3 allennlp_script_V2_14april2021.py
  python3 linguakit_script_14april.py
  python3 DBPEDIA_NER_V7_14april2021.py

  # Uncomment this line if Gazetteer processing is applicable
  # python3 gazetteer1_script.py

  # Combining outputs
  python3 script_join_v4AT_17april2021.py

  # Final decision and BIO column creation
  python3 Decision_V3_19april2021.py

  # Graphical representation
  python3 GraphicRepresentationV3-16april2021.py

  # Generating paper-related results
  python3 ResultPaperSLATE_V1_16april2021.py

  # Save outputs to a specific directory
  local OUTPUT_DIR="${DEST_DIR}${name}/"
  mkdir -p "$OUTPUT_DIR"

  # Copy relevant output files
  cp output.* "$OUTPUT_DIR"
  cp input.txt "$OUTPUT_DIR"
  cp output-*.* "$OUTPUT_DIR"
  cp crosstab*.* "$OUTPUT_DIR"
  cp debug*.* "$OUTPUT_DIR"

  echo "Completed processing for $filename. Outputs saved to $OUTPUT_DIR"

  # Clean up temporary files
  rm -f output.*
  rm -f input.txt
  rm -f aux1.txt
}

export -f process_file  # Export the function so GNU parallel can use it

# Get total number of files to process
FILES=($(find "$SOURCE_DIR" -name "*.txt"))
TOTAL_FILES=${#FILES[@]}

# Process files in parallel with progress tracking
printf "%d files found. Starting processing...\n" "$TOTAL_FILES"

seq 1 "$TOTAL_FILES" | parallel --jobs 4 \
  process_file "${FILES[@]}" ::: {1..${#FILES[@]}} ::: "$TOTAL_FILES"

# Final step: Copy all processed files to the destination directory
cp "$SOURCE_DIR"/*.txt "$DEST_DIR"
echo "All processed files have been copied to $DEST_DIR"