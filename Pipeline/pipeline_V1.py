import subprocess
import time
from datetime import datetime

# List of scripts to run (you can use full paths if needed)
scripts = [#"./INITIAL_NERS/JOIN_NERS.py", 
           "./INITIAL_NERS/CREATE_DECISION_FILE.py",
           "./INITIAL_NERS/REARRANGE_DECISION_TO_BERT.py",
           "./ENTITY_DETECTOR_BY_BERT/STEP1_CRIA_E_TREINA_MODELO.py"
          ]

# Log file to write execution info
log_file = "script_execution_log.txt"

with open(log_file, "w", encoding="utf-8") as log:
    for script in scripts:
        start_time = time.time()
        start_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log.write(f"Starting {script} at {start_str}\n")

        try:
            subprocess.run(["python", script], check=True)
        except subprocess.CalledProcessError as e:
            log.write(f"Error while running {script}: {e}\n")

        end_time = time.time()
        end_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        duration = end_time - start_time

        log.write(f"Finished {script} at {end_str}\n")
        log.write(f"Processing time: {duration:.2f} seconds\n")
        log.write("-" * 40 + "\n")

print(f"All scripts completed. Log saved to {log_file}")
