#!/bin/bash

# Directory containing the files
SOURCE_DIR="/home/ematos/phd/NER_COMPLETE_SYSTEM_V1/INPUT_DATASET/"
DEST_DIR="/home/ematos/phd/NER_COMPLETE_SYSTEM_V1/OUTPUT_RESULTS/"

# Max number of parallel jobs
MAX_JOBS=1
job_count=0  # Counter to keep track of running jobs

# Function to retry operations with delay and exponential backoff
fetch_with_retry() {
  local cmd="$1"
  local retries=5        # Number of retries
  local delay=5          # Base delay in seconds
  local backoff=2        # Exponential backoff factor
  local attempt=1

  while (( attempt <= retries )); do
    # Execute the command
    eval "$cmd"
    if [ $? -eq 0 ]; then
      return 0  # Success
    else
      echo "[Attempt $attempt/$retries] Failed: $cmd"
    fi

    # Check if this is the last attempt
    if (( attempt == retries )); then
      echo "Max retries reached. Skipping: $cmd"
      return 1
    fi

    # Calculate and wait for the next retry
    sleep_time=$(( delay * (backoff ** (attempt - 1)) ))
    echo "Retrying in $sleep_time seconds..."
    sleep "$sleep_time"

    (( attempt++ ))
  done
}

# Function to process each file
process_file() {
  local file="$1"
  local filename=$(basename -- "$file")
  local name="${filename%.*}"
  
  echo "Processing: $name ($filename)"
  
  # Copy the file to the current directory
  cp "$file" "$filename"
  
  # Process the file with retries
  fetch_with_retry "./process_phase.sh \"$name\""
  
  # Clean up output.* and input.txt files
  rm -f output.*
  rm -f input.txt
}

# Export the function so it can be used by subshells
export -f fetch_with_retry
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
  if ((job_count >= MAX_JOBS )); then
    wait
    job_count=0
  fi
done

# Wait for any remaining background jobs to complete
wait

# Copy processed files to the destination directory
cp *.txt "$DEST_DIR/"

echo "All files processed and results saved to $DEST_DIR"