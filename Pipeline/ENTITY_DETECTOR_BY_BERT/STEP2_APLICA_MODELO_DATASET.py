#!/usr/bin/env python
# coding: utf-8

# ###   Input File = /home/ematos/phd/pt/INITIAL_NERS/OUTPUT_RESULTS/concatenated_output_sentences.csv
# ###   Output File = /home/ematos/phd/pt/ENTITY_DETECTOR_BY_BERT/OUTPUT_RESULTS/output_bert_text_city.csv
# ###   Output File 2 = /home/ematos/phd/pt/MAPPER_ENTITY2CLASSES/data/output_bert_text_city.csv

# In[1]:


import pandas as pd
import numpy as np
from tqdm import tqdm, trange
import sys


# In[2]:


import os

working_dir = os.getcwd()
print(working_dir)

parts = working_dir.split("/")
#if parts[-1] == "pipeline":
if parts[-1].startswith("pipeline"): ### BUG SOLVED MAY 2025
    pipeline = working_dir
else:
    pipeline ="/".join(parts[:-1])
print(pipeline)


# In[3]:


sys.path.append(pipeline)

from config.configsite import * # basedir,inputdir,outputdir


# In[29]:


#______________________________________________________________________________________
base_model = f"bert_{dataset}"
model = 'Bert_text_entity_19032025.h5'
#output_model = '/home/ematos/phd/pt/ENTITY_DETECTOR_BY_BERT/OUTPUT_RESULTS'
output_model = pipeline+"ENTITY_DETECTOR_BY_BERT/"+resultsdir  #+"/"
print(f"output_model = {output_model}") # new AT april 2025


# In[30]:


#filename2='/home/ematos/phd/pt/INITIAL_NERS/OUTPUT_RESULTS/concatenated_output_sentences.csv'
filename2= pipeline+dataset+"_PROCESSED/"+resultsdir+"/concatenated_output_sentences_v2.csv"
filename1=f"{output_model}/{model}"
print(f"filename1 ={filename1}")
print(f"filename2 ={filename2}")

output1 = f'{output_model}output_{base_model}.csv'
output2 = f'{output_model}output_{base_model}.csv'

# AT acho melhor não ter 2 outputs... mais fácil de manter consistência 
#  POLITICA ADOTADA: outputs ficam no step em que são gerados
#output2 = f'/home/ematos/phd/pt/MAPPER_ENTITY2CLASSES/data/output_{base_model}.csv'

print(f"output1 ={output1}")


# In[31]:


################### Nome que persiste
resp=f"{output_model}/result_{base_model}.csv"
print(f"resp ={resp}")


# In[32]:


import codecs
teste=[]
#f = codecs.open('wikiner.txt', 'r','utf8')
#f = codecs.open('CDMini.txt', 'r','utf8')
#test = f.read()
#f.close()


# # Q++  onde se cria ?
# ### filename2 =/home/ematos/ateixeira/pipeline/ENTITY_DETECTOR_BY_BERT/OUTPUT_RESULTS//concatenated_output_sentences.csv
# 
# A original do Emanuel está em 
# http://localhost:8889/lab/tree/phd/pt/INITIAL_NERS/OUTPUT_RESULTS/concatenated_output_sentences.csv
# 

# In[33]:


test = pd.read_csv(filename2, encoding="utf8").ffill()


# In[34]:


test


# In[35]:


import math
n=len(test)
m=math.ceil(n/1000)


# In[36]:


#test=test[15000:30000]


# In[37]:


n=len(test)


# In[38]:


n


# In[39]:


test.rename(columns={'line_number':'linha','text':'Sentence'},inplace=True)


# In[40]:


########################################################################
import transformers
from transformers import BertForTokenClassification 
from torch.optim import AdamW

transformers.__version__
##########################################################################


# In[41]:


###################################################
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertConfig

#from keras.preprocessing.sequence import pad_sequences
# NEW VERSION 2 may 2025
from tensorflow.keras.preprocessing.sequence import pad_sequences

from sklearn.model_selection import train_test_split

torch.__version__


# In[42]:


MAX_LEN = 75
bs = 32
#####################################################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()

torch.cuda.get_device_name(0)


# In[43]:


tokenizer = BertTokenizer.from_pretrained(filename1, do_lower_case=False)


# In[44]:


tokenizer


# In[45]:


model = BertForTokenClassification.from_pretrained(
    filename1,
    #num_labels=len(teste),
    output_attentions = False,
    output_hidden_states = False
)


# In[46]:


model.cuda();


# In[47]:


tag_values = []
with open(f"{filename1}/tag.txt", "r") as f:
  for line in f:
    tag_values.append(line.strip())


# In[48]:


tag_values


# In[49]:


#####################################################
def tokenize_and_preserve_labels(sentence, text_labels):
    tokenized_sentence = []
    labels = []

    for word, label in zip(sentence, text_labels):

        # Tokenize the word and count # of subwords the word is broken into
        tokenized_word = tokenizer.tokenize(word)
        n_subwords = len(tokenized_word)

        # Add the tokenized word to the final tokenized word list
        tokenized_sentence.extend(tokenized_word)

        # Add the same label to the new list of labels `n_subwords` times
        labels.extend([label] * n_subwords)

    return tokenized_sentence, labels
###########################################################################


# In[60]:


from tqdm import tqdm

debug = True
debug2 = False

# Open the output file and write the header once
output_filename = f"{output_model}/output_{base_model}.csv"
with open(output_filename, "w", encoding="utf-8") as f:
    f.write("id_file\tlinha\tnword\tWORD\tBIO-BERT\n")  # Write the header

# Function to split long text into chunks of 512 tokens
def split_into_chunks(text, tokenizer, max_length=512):
    tokens = tokenizer.encode(text, add_special_tokens=False)
    return [tokens[i:i + max_length] for i in range(0, len(tokens), max_length)]

# Loop through the rows with tqdm

for j in tqdm(range(n), desc="Processing rows"):
    test_sentence = test.iloc[j, 2]
    linha = test.iloc[j, 0]
    id_file = test.iloc[j, 1]
    
    # Reinicia contador de palavras por sentença
    nword = 0
    
    # Tokenize and split long text into chunks
    tokenized_chunks = split_into_chunks(test_sentence, tokenizer)

    # Process each chunk separately
    for chunk in tokenized_chunks:
        # Convert chunk to tensor and move to GPU
        input_ids = torch.tensor([chunk]).cuda()
        
        with torch.no_grad():
            output = model(input_ids)
        
        label_indices = np.argmax(output[0].to('cpu').numpy(), axis=2)

        # Join BPE split tokens
        tokens = tokenizer.convert_ids_to_tokens(input_ids.to('cpu').numpy()[0])
        new_tokens, new_labels = [], []

        if debug2:
            print(f"tokens = {tokens}")
        
        for token, label_idx in zip(tokens, label_indices[0]):
            if token.startswith("##"):
                if debug2:
                    print(f"token = {token}, label_idx = {label_idx}")
                try:
                    new_tokens[-1] = new_tokens[-1] + token[2:]  ## ??? parece adicionar ao ultimo em new_tokens as letras de 2 ao fim do novo toke
                except  Exception as e:
                    print(f"EXCEPTION! PROBLEMS - {e}")
                    print(f"token = {token}, label_idx = {label_idx}")
                    continue
            else:
                new_labels.append(tag_values[label_idx])
                new_tokens.append(token)
                if debug2:
                    print(f"new_tokens = {new_tokens[-3:]}")

        # Write each token-label pair to the file immediately
        print(f"Saving to {output_filename}")
        with open(output_filename, "a", encoding="utf-8") as f:
            for token, label in zip(new_labels, new_tokens):
                nword += 1   # Incrementa número da palavra dentro da linha
                f.write(f"{id_file}\t{linha}\t{nword}\t{label}\t{token}\n")
                if debug:
                    print(f"|{id_file}|{linha}|{nword}|{label}|{token}|")
                


# In[61]:


print(f"reading from = {output1}")


# In[65]:


#with open(output1, "r", errors="replace") as f:
with open(output1, "r") as f:
    
    lines = f.readlines()

lines


# In[66]:


with open(output1, "r", errors="replace") as f:
    lines = f.readlines()

output1_clean = f'{output_model}cleaned_{base_model}.csv'
with open(output1_clean, "w") as f:
    f.writelines(lines)

df = pd.read_csv(output1_clean, on_bad_lines='skip')


# In[67]:


#df = pd.read_csv(output1,sep="\t")
import csv
#df = pd.read_csv(output1, quoting=csv.QUOTE_ALL, sep="\t",engine='python')

#df = pd.read_csv(output1, engine='python', sep="\t") #, quotechar='"', escapechar='\\')


# In[68]:


df.to_csv(output2,sep="\t",index = False)


# In[ ]:





# In[ ]:




