#!/bin/bash

# Initialize Conda for the script. This should be at the very beginning.
# The exact path might vary. You can find it by running 'conda info --base'
# and then looking for 'etc/profile.d/conda.sh' within that base path.
# A common location is ~/miniconda3/etc/profile.d/conda.sh or ~/anaconda3/etc/profile.d/conda.sh
source "$(conda info --base)/etc/profile.d/conda.sh"

# Function to activate a conda environment and run a command
run_in_env() {
  local env_name="$1"
  local command="$2"
  # Use a subshell to ensure environment changes are localized
  (
    conda activate "$env_name" && \
    eval "$command"
  )
}

# Deactivate any active conda environment to start fresh (now safe to do)
conda deactivate

echo "--- STARTING PROCESS ---"

# --- INITIAL NERS (Parallel Execution) ---
echo "INITIAL NERS - Running in parallel: allennlp, linguakit, dbpedia"

# Run the first three scripts in parallel. The & sends the process to the background.
run_in_env allenlp_env "python -u INITIAL_NERS/script_allennlp.py" &
run_in_env allenlp_env "python -u INITIAL_NERS/script_linguakit.py" &
run_in_env allenlp_env "python -u INITIAL_NERS/script_dbpedia.py" &

# Wait for all background jobs to complete before proceeding
wait

echo "INITIAL NERS - All parallel scripts completed."

# --- JOIN RESULTS OF INITIAL NERS (Sequential) ---
echo "JOIN RESULTS OF INITIAL NERS"
run_in_env allenlp_env "python -u INITIAL_NERS/JOIN_NERS.py | grep ERROR"
run_in_env allenlp_env "python -u INITIAL_NERS/CREATE_DECISION_FILE.py"
run_in_env allenlp_env "python -u INITIAL_NERS/REARRANGE_DECISION_TO_BERT.py"

# --- BERT NER (Sequential) ---
echo "BERT NER"
echo "1"
run_in_env torch-gpu "python -u ENTITY_DETECTOR_BY_BERT/STEP1_CRIA_E_TREINA_MODELO.py"
echo "2"
run_in_env torch-gpu "python -u ENTITY_DETECTOR_BY_BERT/join_sentences.py"
echo "3"
run_in_env torch-gpu "python -u ENTITY_DETECTOR_BY_BERT/STEP2_APLICA_MODELO_DATASET.py"
echo "4"
run_in_env torch-gpu "python -u ENTITY_DETECTOR_BY_BERT/STEP3_JOIN_BERT_INITIAL_NERS.py"

# --- MAPPER_ENTITY2CLASSES (Sequential) ---
echo "MAPPER"
echo "1"
run_in_env allenlp_env "python -u MAPPER_ENTITY2CLASSES/STEP1_MAPPER_NERJOINED.py"
echo "2"
run_in_env torch-gpu "python -u MAPPER_ENTITY2CLASSES/STEP2_CRIA_CLASSES-Copy1.py"

# --- REBEL (Sequential) ---
echo "REBEL"
echo "REBEL 1"
run_in_env torch-gpu "python -u REBEL_ENTITY2CLASSES/STEP1_read_dataset_2025.py"
echo "REBEL 2"
run_in_env torch-gpu "python -u REBEL_ENTITY2CLASSES/STEP2_rebel_train_2025.py"
echo "REBEL 3"
run_in_env torch-gpu "python -u REBEL_ENTITY2CLASSES/STEP3_mapper_rebel_train_2025.py"
echo "REBEL 4"
run_in_env phd "python -u REBEL_ENTITY2CLASSES/STEP4_mapper2classes.py"
echo "REBEL 5"
run_in_env torch-gpu "python -u REBEL_ENTITY2CLASSES/STEP5_join_rebel_results.py"

# --- JOIN MAPPER with REBEL (Sequential) ---
echo "JOINING MAPPER with REBEL"
echo "STEP1"
run_in_env torch-gpu "python -u JOIN_MAPPER_REBEL/STEP1_JOIN_MAPPER_REBEL.py"
echo "STEP2"
run_in_env phd "python -u JOIN_MAPPER_REBEL/STEP2_CRIA_CLASSES.py"

# --- NEW INDOMAIN NER FLAIR (Sequential) ---
echo "NEW NER FLAIR"
echo "STEP 1"
run_in_env torch-gpu "python -u NEW_INDOMAIN_NER_FLAIR/STEP01_ajuste_BIO.py"
echo "STEP 2"
run_in_env torch-gpu "python -u NEW_INDOMAIN_NER_FLAIR/STEP02_split_TRAIN_TEST_flair_2025.py"
echo "STEP 3"
run_in_env torch-gpu "python -u NEW_INDOMAIN_NER_FLAIR/STEP03_process_our_new_dataset_v6_29abr2025.py"

# --- NEW INDOMAIN NER LLM (Sequential) ---
echo "NEW NER LLM"
run_in_env phd "python -u NEW_INDOMAIN_NER_LLM/STEP00_create_train_test_xml_sabiazinho_v4.py"

echo "--- SCRIPT COMPLETED ---"