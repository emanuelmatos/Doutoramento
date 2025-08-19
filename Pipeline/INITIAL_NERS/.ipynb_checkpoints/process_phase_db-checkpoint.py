import os
import shutil
import subprocess
import sys

def run_command(command, success_message, failure_message):
    try:
        result = subprocess.run(command, check=True, shell=True)
        print(success_message)
        return result.returncode
    except subprocess.CalledProcessError:
        print(failure_message)
        return 1


if len(sys.argv) > 1 and sys.argv[1] == '-f':
    file_name = 'Abrantes.txt'
    dir_name = 'Abrantes'
    print(f"Filename is set to {file_name}")
else:
    file_name = f"{sys.argv[1]}.txt"
    dir_name = sys.argv[1]

def main():
    if len(sys.argv) < 2:
        print("Usage: python script.py <filename_without_extension>")
        sys.exit(1)

    filename = file_name
    dirname = dir_name

    print(f"Processing {filename} ...........................................................................")

    # Copy file to be processed to aux1.txt
    shutil.copy(filename, "aux1.txt")

    # Preprocess to solve some formatting issues
    print("Preprocessing input file...")
    run_command("python preprocess.py", "Preprocessing completed successfully.", "Preprocessing failed! Continuing to next steps.")

  
         # NER 3 - DBPEDIA
    print("Running NER 3 - DBPEDIA...")
    if not os.path.exists("output.dbpedia"):
        run_command(f"python DBPEDIA_NER_V7_19dec2024.py {filename}", 
                    "NER 3 - DBPEDIA completed successfully.", 
                    "NER 3 - DBPEDIA failed! Skipping to next step.")
    else:
        print("NER 3 - DBPEDIA output file already exists. Skipping.")


    # Create a directory with the same name as the text file and move all output files there
    print(f"Saving output files to directory: {dirname}")
    os.makedirs(dirname, exist_ok=True)

    for pattern in ["output.*", "input.txt", "output-*.*", "crosstab*.*", "debug*.*"]:
        for file in [f for f in os.listdir('.') if os.path.isfile(f) and f.startswith(pattern.split('*')[0])]:
            shutil.move(file, os.path.join(dirname, file))

    print(f"Processing completed successfully (with potential skipped steps)! {filename} ...........................................................................")

if __name__ == "__main__":
    main()