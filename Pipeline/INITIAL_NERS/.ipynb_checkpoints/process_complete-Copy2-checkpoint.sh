#!/bin/bash

# Directory containing the files
SOURCE_DIR="/home/ematos/phd/NER_COMPLETE_SYSTEM_V1/INPUT_DATASET/"
DEST_DIR="/home/ematos/phd/NER_COMPLETE_SYSTEM_V1/OUTPUT_RESULTS/"

# Max number of parallel jobs
MAX_JOBS=4

# Counter to keep track of running jobs
job_count=0

# Function to process each file
process_file() {
  local file="$1"
  local filename=$(basename -- "$file")
  local name="${filename%.*}"
  
  echo "Processing: $name ($filename)"
  
  # Copy the file to the current directory
  cp "$file" "$filename"
  
  # Process the file
  ./process_phase.sh "$name"
  
  # Clean up output.* and input.txt files
  rm -f output.*
  rm -f input.txt
}

# Export the function so it can be used by subshells
export -f process_file

# Export necessary variables for subshells
export SOURCE_DIR
export DEST_DIR

# Loop through all .txt files in the source directory
for file in "$SOURCE_DIR"/*.txt; do
  process_file "$file" &
  
  # Increment the job counter
  ((job_count++))
  
  # Wait if the number of jobs reaches the limit
  if ((job_count >= MAX_JOBS)); then
    wait
    job_count=0
  fi
done

# Wait for any remaining background jobs to complete
wait

# Copy files to the destination directory
cp *.txt "$DEST_DIR/"

echo "All files processed and results saved to $DEST_DIR"