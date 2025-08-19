#!/usr/bin/env python3

import os
import csv
import shutil
import time
from multiprocessing import Pool

# Directories
SOURCE_DIR = "/Volumes/hd_ext/phd/NER_COMPLETE_SYSTEM_V1/INPUT_DATASET/"
DEST_DIR = "/Volumes/hd_ext/phd/NER_COMPLETE_SYSTEM_V1/OUTPUT_RESULTS/"
CSV_FILE = "process.csv"
PROCESSED_LIST = "list_processed.csv"
MAX_PROCESSES = 2  # Limit the number of simultaneous processes

# Ensure the processed list file exists
if not os.path.isfile(PROCESSED_LIST):
    with open(PROCESSED_LIST, "w") as file:
        file.write("Filename,Base Name\n")

# Function to process a single file
def process_file(row):
    filename, name = row
    print(f"Processing file: {filename} with base name: {name} ...")

    # Ensure the output directory exists for this file
    output_dir = os.path.join(DEST_DIR, name)
    os.makedirs(output_dir, exist_ok=True)

    # Copy the file to the current directory for processing
    src_file = os.path.join(SOURCE_DIR, filename)
    temp_file = filename
    shutil.copy(src_file, temp_file)

    try:
        # Retry mechanism
        max_retries = 4
        delay = 3  # Initial delay in seconds

        for attempt in range(max_retries):
            try:
                # Run the processing script (this is likely where HTTPError occurs)
                os.system(f"./process_phase.sh {name}")
                break  # Exit the retry loop if successful
            except Exception as e:
                print(f"Error on attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(delay)
                    delay *= 2  # Exponential backoff
                else:
                    print(f"Max retries reached for {filename}. Skipping.")
                    with open("error_log.txt", "a") as log_file:
                        log_file.write(f"{filename}: Max retries reached. {e}\n")
                    raise

        # Move all output files to the specific output directory
        for output_file in ["output.*", "input.txt"]:
            for file in os.listdir():
                if file.startswith(output_file.split('.')[0]):
                    shutil.move(file, output_dir)

    except Exception as e:
        print(f"Error processing {filename}: {e}")
        with open("error_log.txt", "a") as log_file:
            log_file.write(f"{filename}: {e}\n")

    finally:
        # Clean up temporary files
        for temp in ["output.*", "input.txt"]:
            for file in os.listdir():
                if file.startswith(temp.split('.')[0]):
                    os.remove(file)

    # Log the processed file
    with open(PROCESSED_LIST, "a") as file:
        writer = csv.writer(file)
        writer.writerow([filename, name])

    print(f"Finished processing file: {filename} with base name: {name}")

# Main function
def main():
    print("Reading process.csv and preparing to process files...")

    # Read the CSV file, skip the header
    with open(CSV_FILE, "r") as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header
        rows = list(reader)

    # Process files in parallel using a limited number of processes
    with Pool(processes=MAX_PROCESSES) as pool:
        pool.map(process_file, rows)

    print(f"All files processed. Results saved to {DEST_DIR}.")
    print(f"Processed files logged in {PROCESSED_LIST}.")

if __name__ == "__main__":
    main()