#!/usr/bin/env python
# coding: utf-8

# #### Script for evaluating entity recognition prompts using the MariTalk model (sabiazim-3).
# Applies few-shot prompting to classify entities in Portuguese text.

# # imports

# In[1]:


#!pip install openpyxl


# In[2]:


# === Imports ===
import os
import time
import re
import pandas as pd
import numpy as np
 
 
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# # CONFIG OF PROCESS

# In[3]:


# number of examples to be used in prompts per class
N_EXAMPLES_PER_CLASS = 3

MINIMO_TAMANHO_FRASE = 30
MINIMO_TAMANHO_ENTITY = 3


# # get info from config file

# In[ ]:





# In[4]:


working_dir = os.getcwd()
print(working_dir)

parts = working_dir.split("/")

if parts[-1].startswith("pipeline"): ### BUG SOLVED MAY 2025
    pipeline = working_dir
else:
    pipeline ="/".join(parts[:-1])
print(pipeline)


# In[5]:


import sys 
base = pipeline
sys.path.append(base)

from config.configsite import *

#print(f"filesdir = {filesdir}") # new AT april 2025
print(f"pipeline = {pipeline}") # new AT april 2025


# In[6]:


# directory with file concatenated_output_sentences....
inputdir=f"{pipeline}{dataset}_PROCESSED/"+resultsdir   
print(f"input dir = {inputdir}") # new AT may 2025


# # read concatenated sentences  (to create df_train_set)

# ### DEBUG
# path_data = "/home/ematos/phd/pt/NEW_INDOMAIN_NER_LLM/INPUT_DATASET/"
# extension = ".parquet"
# BASE_FILE = 'concatenated_output_sentences'+extension
# if extension == ".parquet":
#     em_df_train_set = pd.read_parquet(path_data+BASE_FILE)
# 
# em_df_train_set.head(3)

# In[7]:


path_data = inputdir
extension = ".csv"
BASE_FILE = 'concatenated_output_sentences_v2'+extension
if extension == ".parquet":
    df_train_set = pd.read_parquet(path_data+BASE_FILE)
if extension == ".csv":
    df_train_set = pd.read_csv(path_data+BASE_FILE)
print(df_train_set.columns)

print(df_train_set.head(3))


# In[8]:


print(df_train_set.sample(5))


# In[9]:


outputdir=pipeline+"PROCESS_TEST_SET_WITH_NER_LLM/"+resultsdir
print(f"output dir = {outputdir}") # new AT may 2025

os.makedirs(outputdir, exist_ok=True)


# In[30]:


#rom sklearn.model_selection import train_test_split


#est_percentage = 30 # use to configure

# Split the data into 70% train and 30% test
#train_df, test_df = train_test_split(df_train_set, 
#                                     test_size=test_percentage /100, 
#                                     random_state=42)

train_df = df_train_set

# Verify the sizes
print(f"Training set size: {len(train_df)} samples ({len(train_df)/len(df_train_set)*100:.1f}%)")
#print(f"Test set size: {len(test_df)} samples ({len(test_df)/len(df_train_set)*100:.1f}%)")

print(train_df.head())


# ### save train data to CSV files

# In[11]:


filename = outputdir+"train.csv"
print(f"Saving to {filename}")
train_df.to_csv(filename)

#filename = outputdir+"test.csv"
#print(f"Saving to {filename}")
#test_df.to_csv(filename)


# # usar LLM

# ### Imports e configuração do Maritalk

# In[12]:


import maritalk
from maritalk.resources.api import MaritalkHTTPError  # API exception handling

# === Constants & Configurations ===
API_KEY = "655f5dd150edc32693e4b855_fadaf137f7b2f01c"
MODEL_NAME = "sabiazim-3"




# ### definição de input e output para LLM

# #### anotações anteriores NER FLAIR para usar como exemplos nos prompts

# In[13]:


#path_data = "/home/ematos/phd/pt/NEW_INDOMAIN_NER_LLM/INPUT_DATASET/"
filename_input = "results_top10_adjst.csv"  ######  agora vem do output do NER FLAIR
path_data=pipeline+"NEW_INDOMAIN_NER_FLAIR/"+resultsdir
print(f"input FLAIR tagged examples = {path_data}{filename_input}") # new AT may 2025


# In[14]:


#OUTPUT_DIR = '/home/ematos/phd/pt/NEW_INDOMAIN_NER_LLM/OUTPUT_RESULTS/maritalk/'
outputdir =pipeline+"PROCESS_TEST_SET_WITH_NER_LLM/"+resultsdir
outputdir_maritalk =pipeline+"PROCESS_TEST_SET_WITH_NER_LLM/"+resultsdir+"maritalk"


print(f"Saving to {outputdir}")
os.makedirs(outputdir, exist_ok=True)


# In[15]:


TEST_FILE = 'test.csv' ######## Ver de onde vem e deixar escrito
TRAIN_FILE = 'train.csv'


# In[16]:


################################################################################################
OUTPUT_FILE = f'{outputdir}XML_2PerClass_10class_sabiazinho3_{dataset}_pt_512_20250615.txt'
print(f"Output to {OUTPUT_FILE}")


DEBUG = True


# ### create model and configure it

# In[17]:


# === Initialize MariTalk model ===
model = maritalk.MariTalk(key=API_KEY, model=MODEL_NAME)

# === Configure request retries (optional, improves stability) ===
session = requests.Session()
retry = Retry(connect=3, backoff_factor=0.5)
adapter = HTTPAdapter(max_retries=retry)
session.mount('http://', adapter)
session.mount('https://', adapter)


# # Load data 

# In[18]:


# === Load Data ===
#os.chdir(path_data)

aux = path_data+filename_input
print(f"Reading examples from {aux}")
word_tag_df = (
    pd.read_csv(aux, sep='\t')[['WORD', 'MAPPER10', 'line',"file_id"]]
    .rename(columns={'MAPPER10': 'tag','WORD': 'word','file_id': 'filename'})
)

print(word_tag_df.head(3))


# In[19]:


print(word_tag_df.sample(3))


# # DUVIDA - usar tag column or BIO column ???

# In[20]:


# em_word_tag_df.head(3)


# ### Extract unique entity tags ===

# In[21]:


#entity_tags = np.setdiff1d(word_tag_df['tag'].unique(), ['O'])
entity_tags = np.setdiff1d(
    word_tag_df['tag'].dropna().astype(str).unique(),
    ['O']
)
entity_tags_df = pd.DataFrame(entity_tags, columns=['tag'])
print(entity_tags_df.head(3))


# ### === Filter only entity-tagged words ===

# In[22]:


# === Filter only entity-tagged words ===
df_entities = (
    word_tag_df.merge(entity_tags_df, on='tag')
    .assign(tag2=lambda df: df['tag'])
    .sort_values("line")
    .assign(idx=lambda df: np.arange(len(df)))
)

print(df_entities.head(3))


# # Load sentence datasets

# ## Train set

# In[ ]:


# === Load sentence datasets === =============================================
train_sentences = (
    pd.read_csv(outputdir+TRAIN_FILE)
    .sort_values("line_number")
    .assign(idx=lambda df: np.arange(len(df)))
)
train_sentences.rename(columns={"line_number": "line","text":"sentence"}, inplace=True)


# ## test data

# #### get sentences of test set

# In[31]:


path_data = inputdir
extension = ".csv"
BASE_FILE = 'concatenated_test_sentences_v2'+extension
if extension == ".parquet":
    df_test_set = pd.read_parquet(path_data+BASE_FILE)
if extension == ".csv":
    df_test_set = pd.read_csv(path_data+BASE_FILE)
print(df_test_set.columns)

print(df_test_set.head(3))


# ###  save to test.csv
# 
# 

# In[32]:


filename = outputdir+"test.csv"
print(f"Saving to {filename}")
df_test_set.to_csv(filename)


# In[33]:


test_sentences = (
    pd.read_csv(outputdir+TEST_FILE, usecols=["line_number", "text"])
    .sort_values("line_number")
    .assign(idx=lambda df: np.arange(len(df)))
)

test_sentences.rename(columns={"line_number": "line","text":"sentence"}, inplace=True)
print(f"Treino = {len(train_sentences)}, Teste = {len(test_sentences)}")

print(test_sentences.head())


# ### Create tag matrix at sentence level

# In[34]:


#################################################################################

# === Create tag matrix at sentence level ===
df_sent_with_tags = train_sentences.set_index("line")

# AT ADDED 4 june
#condicao = len(df_sent_with_tags["sentence"]) > MINIMO_TAMANHO_FRASE
condicao = df_sent_with_tags['sentence'].str.len() > MINIMO_TAMANHO_FRASE
df_sent_with_tags = df_sent_with_tags[condicao]

# AT ADDED 4 june
df_sent_with_tags = df_sent_with_tags.sort_values("filename")

print(df_sent_with_tags.head())


# In[35]:


df_entities.head()


# # entities processing

# In[36]:


print(len(df_entities))
 
# AT ADDED 4 june
#condicao = len(df_sent_with_tags["sentence"]) > MINIMO_TAMANHO_FRASE
condicao = df_entities['word'].str.len() > MINIMO_TAMANHO_ENTITY
df_entities = df_entities[condicao]


print(len(df_entities))

condicao2 = df_entities['word'].str.len() <= MINIMO_TAMANHO_ENTITY
aux = df_entities[condicao2]
aux
 


# # NEW 4 JUNHO - deleter very small words tagged
# 
# 

# In[ ]:





# In[37]:


# STEP - create columns for all the classes / tags AND initialize with 0
for tag in entity_tags:
    print(f"tags = {tag}")
    
    df_sent_with_tags[tag] = 0

print(df_sent_with_tags)


# In[38]:


for _, row in df_entities.iterrows():
    #print(f"ROW =\n {row}")

    # same line
    condicao1 = df_sent_with_tags.index == row['line']
    # same file
    condicao2 = df_sent_with_tags["filename"] == row['filename']
    aux = df_sent_with_tags[(condicao1) &  (condicao2)]

    #print(dict(aux))
    if len(aux) > 0:
        df_sent_with_tags.loc[(condicao1) &  (condicao2), row['tag2'] ] = 1
    
    #files_match = row['filename'] == df_sent_with_tags["filename"]            # AT ADDED 4 de june
    #if (row['line'] in df_sent_with_tags.index) and (files_match):
    #    df_sent_with_tags.loc[row['line'], row['tag2']] = 1

    #print(aux.index)

    # V1
    #if len(aux) > 0:
    #    df_sent_with_tags.loc[row['line'], row['tag2']] = 1

#df_sent_with_tags.drop("filename", axis=1, inplace=True)
print(df_sent_with_tags.head(3))


# In[39]:


df_sent_with_tags['SUM'] = df_sent_with_tags.iloc[:, 4:].sum(axis=1)
df_sent_with_tags.sort_values("SUM",ascending = False).head()


# In[40]:


df_sent_with_tags.to_csv("matrix.csv", encoding = "utf8")


# ### Utility Functions

# In[41]:


def get_random_sentence_with_index(df, entity, debug = False, ns = 5):
    """Returns (sentence, line_number) of one random sentence containing the entity."""

    # Passo1 - get ns random lines (that contain entity)
    sample1 = df[df[entity] == 1].sample(ns)
    sample1 = sample1.reset_index()

    # Passo 2 - sort by number of tags in sentence (in SUM)
    aux = sample1.sort_values("SUM",ascending=False)
    #print(aux)
    sample = aux.iloc[0]
    

    if debug:
        print(f"{entity} =>\n{sample}")

    sentence = sample.sentence #iloc[0, 2]
    line_number = int(sample.line) #int (sample.index[0])   #int(sample.iloc[0,3])  # WAS sample.index[0]    !!IMPORTANT, 3 junho 2025
    filename = sample.filename #sample.iloc[0, 1]
    if debug:
        #print(sample)
        print(sentence)

    #print(line_number)
    
    
    return sentence,  line_number, filename # (sentence, line_number)


# In[42]:


example =get_random_sentence_with_index(df_sent_with_tags, entity_tags[0], False, 5)
print(example)


# In[43]:


# === Utility Functions ===
def get_random_sentence_with_index_old(df, entity, debug = False):
    """Returns (sentence, line_number) of one random sentence containing the entity."""
    sample = df[df[entity] == 1].sample(1)

    if debug:
        print(f"{entity} => {sample}")

    sentence = sample.iloc[0, 2]
    line_number = int (sample.index[0])   #int(sample.iloc[0,3])  # WAS sample.index[0]    !!IMPORTANT, 3 junho 2025
    filename = sample.iloc[0, 1]
    if debug:
        #print(sample)
        print(sentence)

    #print(line_number)
    
    
    return sentence,  line_number, filename # (sentence, line_number)
    


# In[ ]:





# # test functions

# In[44]:


example =get_random_sentence_with_index(df_sent_with_tags, entity_tags[0], 1)
print(example)


# In[45]:


example =get_random_sentence_with_index(df_sent_with_tags, entity_tags[1], 1)
print(example)


# In[46]:


def get_sentences_for_all_entities(df, tags, k=1, debug = False):
    """Returns k examples for each entity tag."""
    examples = []
    for tag in tags:
        if debug:
            print(f"TAG = {tag}")
        for _ in range(k):
            try:
                examples.append(get_random_sentence_with_index(df, tag))
            except Exception as e:
                print(f"Problems with {tag}, continuing")
                      
    return examples


# In[47]:


examples = get_sentences_for_all_entities(df_sent_with_tags, entity_tags, 2)
#examples


# In[48]:


def wrap_word_with_tag(text, word, tag):
    """Wrap word with entity tag using regex."""
    pattern = re.compile(rf'\b({re.escape(word)})\b', re.IGNORECASE)
    return pattern.sub(rf'<{tag}>\1</{tag}>', text)

def generate_with_retry(model, prompt, retries=3, delay=5):
    """Retry logic for handling API call failures."""
    for attempt in range(retries):
        try:
            return model.generate(
                prompt,
                chat_mode=False,
                do_sample=False,
                max_tokens=512, ###### TO DO avaliar a literatura era 220 coloquei 512 (frases maiores)
                stopping_tokens=["\n"]
            )
        except MaritalkHTTPError as e:
            if attempt < retries - 1:
                print(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay}s...")
                time.sleep(delay)
            else:
                print("All retries failed.")
                raise


# #### tests das funções

# In[ ]:





# # Inference and Prompting 

# In[49]:


df_sent_with_tags.head(3)


# In[50]:


entity_tags


# In[51]:


df_entities


# In[53]:


print(f"{OUTPUT_FILE}")


# In[54]:


# === Inference and Prompting ===#
#os.chdir(OUTPUT_DIR)

TAMANHO_MINIMO_FRASE = 20
debug2 = False
debug3 = False

with open(OUTPUT_FILE, "w", encoding="utf8") as out:
    out.write("linha\tfrase\tfrase_tag\n")

    for idx, row in test_sentences.iterrows():
        time.sleep(10)  # Respect API rate limits
        linha, frase = row['line'], row['sentence']

        # AT NEW 4 JUNHO
        if len(frase) < TAMANHO_MINIMO_FRASE:
            print("Sentence too short, skipping...")
            continue

        if DEBUG:
            print(f"\n---\n{idx} : Frase = {frase}\nLinha = {linha}\n")

        # Build prompt with k-shot examples
        prompt = (
        "Você receberá frases em português. Marque as entidades presentes em cada frase "
        "usando o formato <TIPO_ENTIDADE>palavra</TIPO_ENTIDADE>. Utilize as mesmas tags "
        "dos exemplos abaixo. Exemplo de entidades: MUNICÍPIO_DO_BRASIL, PESSOA, ORGANIZAÇÃO, etc.\n"
        "Deve juntar as palavras com as mesmas tags, como, por exemplo <CIDADE_GRANDE>São</CIDADE_GRANDE> <CIDADE_GRANDE>Paulo</CIDADE_GRANDE> deve ser <CIDADE_GRANDE>São Paulo</CIDADE_GRANDE>\n"
        )
        # Correct approach (option 1 - swap variables in loop)

        #  
        examples = get_sentences_for_all_entities(df_sent_with_tags, entity_tags, N_EXAMPLES_PER_CLASS)

        #print(f"Examples: {examples}")  # PROBLEM in get_sentences...
        
        total_tokens = 0

        #  
        for sentence, line_number, filename in examples:  # Clear variable names
            sentence = str(sentence).strip()  # Ensure we're working with a string

            if debug2:
                print(f"(1) EXAMPLE SENTENCE (line {line_number}, file {filename}) = |{sentence}|")


            # BUG TO BE SOLVED - considerar que tem de ser também no mesmo ficheiro
            # AT : CHANGED 4 june

            # DEBUG
            if debug2:
                print(f"LOOKING for line {line_number} in {filename} -- {sentence}")
            
            ent_data = (
                df_entities[(df_entities.line == line_number) & (df_entities.filename == filename) & (df_entities.tag != 'O')]
                .drop_duplicates(subset='word')
            )


            # TODO: IMPROVE list of examples to give the small amountof continues IMPORTANT
            if ent_data.empty:
                print("NO DATA... skipping")
                continue

            if debug2:
                print(f"\n(2)ENTITIES TO TAG :\n {ent_data}")  # TODO: check if line_number ok
            
            frase_tag = sentence
            for _, token in ent_data.iterrows():
                word = token['word'].replace('_', ' ')
                tag = token['tag']
                frase_tag = wrap_word_with_tag(frase_tag, word, tag)
            
            if debug3:
                print(f"\n(3)FRASE TAGGED = {frase_tag}")

            
            example_prompt = f"Frase: {sentence}\nFrase com entidade: {frase_tag}\n\n"

            #print(f"Example: {example_prompt}")
            
            total_tokens += len(example_prompt.split())
            if total_tokens >= 8000:
                print("Prompt token limit reached. Skipping remaining examples.")
                break
            prompt += example_prompt

        prompt += f"Frase: {frase}\nFrase com entidade:"

        try:
            answer = generate_with_retry(model, prompt)
            print(f"# {idx}/{len(test_sentences)} | Linha: {linha}\nResposta: {answer['answer']}")
            out.write(f"{linha}\t{frase}\t{answer['answer']}\n")
        except MaritalkHTTPError:
            print(f"Erro ao obter resposta para linha {linha}")


# In[ ]:





# In[ ]:




