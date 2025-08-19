#!/bin/bash

# Usage: ./replicate.sh destination_directory [relative to ..]

SOURCE="."
DEST="../$1"

if [ -z "$SOURCE" ] || [ -z "$DEST" ]; then
  echo "Usage: $0  destination_directory [relative to ..]"
  exit 1
fi

# Ensure source exists
if [ ! -d "$SOURCE" ]; then
  echo "Error: Source directory does not exist."
  exit 1
fi

# Create destination directory if it doesn't exist
mkdir -p "$DEST"

# Use rsync to copy everything except .ipynb files
rsync -av --exclude='*.ipynb' \
  --exclude='*.log' \
   --exclude='*.txt' \
    --exclude='*.csv' \
  --exclude='*.pyc' "$SOURCE/" "$DEST/"

echo "Replication completed, excluding .ipynb files."