
# conda info --envs

# conda init


# conda activate allenlp_env

# FIRST VERSION. Not ok
conda deactivate
conda activate allenlp_env
echo "INITIAL NERS"
python -u INITIAL_NERS/script_allennlp.py 
python -u INITIAL_NERS/script_linguakit.py 
python -u INITIAL_NERS/script_dbpedia_july_2025.py

# join etc
echo "JOIN RESULTS OF INITIAL NERS"
python -u INITIAL_NERS/JOIN_NERS.py | grep ERROR
python -u  INITIAL_NERS/CREATE_DECISION_FILE.py
python -u  INITIAL_NERS/REARRANGE_DECISION_TO_BERT.py

# BERT NER
echo "BERT NER"
conda deactivate
conda activate torch-gpu
echo "1"
python -u ENTITY_DETECTOR_BY_BERT/STEP1_CRIA_E_TREINA_MODELO.py 
echo "2"
python -u ENTITY_DETECTOR_BY_BERT/join_sentences.py
echo "3"
python -u ENTITY_DETECTOR_BY_BERT/STEP2_APLICA_MODELO_DATASET.py 
echo "4"
python -u ENTITY_DETECTOR_BY_BERT/STEP3_JOIN_BERT_INITIAL_NERS.py

# MAPPER_ENTITY2CLASSES
echo "MAPPER"
conda deactivate
conda activate allenlp_env
echo "1"
python -u MAPPER_ENTITY2CLASSES/STEP1_MAPPER_NERJOINED.py 

conda deactivate
conda activate torch-gpu
echo "2"
python -u MAPPER_ENTITY2CLASSES/STEP2_CRIA_CLASSES-Copy1.py


# REBEL 
echo "REBEL"
conda deactivate
conda activate torch-gpu
echo "REBEL 1"
python -u REBEL_ENTITY2CLASSES/STEP1_read_dataset_2025.py
echo "REBEL 2"
python -u REBEL_ENTITY2CLASSES/STEP2_rebel_train_2025.py 
echo "REBEL 3"
python -u REBEL_ENTITY2CLASSES/STEP3_mapper_rebel_train_2025.py 
echo "REBEL 4"
conda deactivate
conda activate phd
python -u REBEL_ENTITY2CLASSES/STEP4_mapper2classes.py 
echo "REBEL 5"
conda deactivate
conda activate torch-gpu
python -u REBEL_ENTITY2CLASSES/STEP5_join_rebel_results.py 
# done


# JOIN MAPPER with REBEL
echo "JOINING MAPPER with REBEL"
conda deactivate
conda activate torch-gpu
echo "STEP1"
python -u JOIN_MAPPER_REBEL/STEP1_JOIN_MAPPER_REBEL.py 
# done
echo "STEP2"
conda deactivate
conda activate phd
python -u JOIN_MAPPER_REBEL/STEP2_CRIA_CLASSES.py 
# done


# NEW INDOMAIN NER FLAIR
echo "NEW NER FLAIR"
conda deactivate
conda activate torch-gpu
echo "STEP 1"
python -u NEW_INDOMAIN_NER_FLAIR/STEP01_ajuste_BIO.py
# done
echo "STEP 2"
python -u NEW_INDOMAIN_NER_FLAIR/STEP02_split_TRAIN_TEST_flair_2025.py
# done
echo "STEP 3"
python -u NEW_INDOMAIN_NER_FLAIR/STEP03_process_our_new_dataset_v6_29abr2025.py
# running


# NEW INDOMAIN NER LLM
echo "NEW NER LLM"
conda deactivate
conda activate phd
python -u NEW_INDOMAIN_NER_LLM/STEP00_create_train_test_xml_sabiazinho_v4.py


# RUN NER with TEST SET


