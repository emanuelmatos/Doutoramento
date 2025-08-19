 conda init
 
# NER FLAIR
echo "PROCESS TEST SET w NER FLAIR"
conda activate torch-gpu
echo "1"
python  PROCESS_TEST_SET_WITH_NER_FLAIR/STEP00_join_sentences.py

echo "2"
python -u PROCESS_TEST_SET_WITH_NER_FLAIR/STEP01_process_v1_14jun2025.py 



# NER LLM
echo "PROCESS TEST SET w NER LLM"
 
echo "1"
conda activate phd
python -u  PROCESS_TEST_SET_WITH_NER_LLM/STEP01_Process_Test_set_by_NER_LLM_sabiazinho_v1.py 

 
echo "2"
conda activate phd
python -u PROCESS_TEST_SET_WITH_NER_LLM/STEP02_Convert_Output_to_BIO_v1.py 

echo "3"
conda activate phd
python -u PROCESS_TEST_SET_WITH_NER_LLM/STEP03_Process_Test_set_by_NER_LLM_zero-shot_sabiazinho_v1.py 

 
echo "4"
conda activate phd
python -u PROCESS_TEST_SET_WITH_NER_LLM/STEP04_Convert_ZERO-SHORT_Output_to_BIO_v1.py 