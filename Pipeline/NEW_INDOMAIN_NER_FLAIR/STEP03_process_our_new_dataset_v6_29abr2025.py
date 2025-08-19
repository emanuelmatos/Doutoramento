#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import random

random.seed(10)
print(random.random())


# ## link para modelos

# https://huggingface.co/models

# In[2]:


#pip install --upgrade flair


# ### config experiment

# In[3]:


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


# In[4]:


import sys 
base = pipeline
sys.path.append(base)

from config.configsite import *

#print(f"filesdir = {filesdir}") # new AT april 2025
print(f"pipeline = {pipeline}") # new AT april 2025


#inputdir=pipeline+"JOIN_MAPPER_REBEL/"+resultsdir
inputdir=pipeline+"NEW_INDOMAIN_NER_FLAIR/"+resultsdir   # AT 24 de maio 2025

print(f"input dir = {inputdir}") # new AT may 2025

outputdir=pipeline+"NEW_INDOMAIN_NER_FLAIR/"+resultsdir
print(f"output dir = {outputdir}") # new AT may 2025

os.makedirs(outputdir, exist_ok=True)


# In[5]:


# tag column to use
#tag_column = "ner1"
tag_column = "ner2"  # 1 class only


# diretory for dataset to be USE - NEW 25 october ---------------
top = 10
train_percent = 60 # can be 25, 50 and 75
test_percent = 30 # can only be 10
dev_percent = 10 # can only be 10

dataset_dir = inputdir+"dataset_bert_and_mapper/"
print(f"input dataset dir = {dataset_dir}") # new AT may 2025


#dataset_dir = f"/home/ematos/phd/pt/NEW_INDOMAIN_NER_FLAIR/data_bert_and_mapper_2025/"
#dataset_dir = f"./train/datawikiner_{top}class_train{train_percent}_dev{dev_percent}_test{test_percent}/"


# In[6]:


# foundation model to be fine tuned
#modelo_base="Geotrend/bert-base-pt-cased"
#link https://huggingface.co/Geotrend/bert-base-pt-cased
modelo_base="vocabtrimmer/xlm-roberta-base-trimmed-pt-60000"
#link https://huggingface.co/vocabtrimmer/xlm-roberta-base-trimmed-pt-60000
print(f"model base = {modelo_base}") 

    
#modelo_base_short="bert-base-pt-cased"
modelo_base_short = "xlm-roberta-base-trimmed-pt-60000"
print(f"model base short = {modelo_base_short}") 

# dir and name for model to be created
#model_created =f'/home/ematos/phd/pt/NEW_INDOMAIN_NER_FLAIR/data_bert_and_mapper_2025'
model_created = inputdir+"dataset_bert_and_mapper/"

#model_created = f'resources/taggers/ner_1class_{top}_{modelo_base_short}_{train_percent}p'


# In[7]:


dataset_dir


# In[8]:


import flair, torch

device = None
if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('mps')


# In[9]:


device


# #  load a custom dataset

# In[10]:


from flair.data import Corpus
from flair.datasets import ColumnCorpus


# Reading a dataset in column format
# In cases you want to train over a sequence labeling dataset that is not in the above list, you can load them with the ColumnCorpus object. Most sequence labeling datasets in NLP use some sort of column format in which each line is a word and each column is one level of linguistic annotation. See for instance this sentence:
# 
# George N B-PER
# 
# Washington N I-PER
# 
# went V O
# 
# to P O
# 
# Washington N B-LOC
# 
# Sam N B-PER
# 
# Houston N I-PER
# 
# stayed V O
# 
# home N O

# The first column is the word itself, the second coarse PoS tags, and the third BIO-annotated NER tags. Empty line separates sentences. To read such a dataset, define the column structure as a dictionary and instantiate a

# ## ver 

# In[11]:


# define columns
columns = {0: 'text', 1: 'ner', 2:'nline'}

# this is the folder in which train, test and dev files reside
data_folder = dataset_dir

# init a corpus using column format, data folder and the names of the train, dev and test files
corpus: Corpus = ColumnCorpus(data_folder, columns,
                              train_file='train.csv',
                              test_file='test.csv',
                              dev_file='dev.csv')


# You can also access a sentence and check out annotations. Lets assume that the training split is read from the example above, then executing these commands

# In[12]:


print(corpus.train[0].to_tagged_string('ner'))


# In[13]:


print(corpus.train[10].to_tagged_string('nline'))


# ## process 

# In[14]:


#from flair.datasets import CONLL_03
from flair.embeddings import WordEmbeddings, FlairEmbeddings, StackedEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer


# ## code below is from https://flairnlp.github.io/docs/tutorial-training/how-to-train-sequence-tagger

# In[15]:


tagger=None
torch.cuda.empty_cache() 


# ##  ver para mais informação
# https://yulianudelman.medium.com/named-entity-recognition-with-flair-4d627b18f5b7#:~:text=from%20flair.embeddings%20import%20TransformerWordEmbeddings%20embeddings%20%3D,TransformerWordEmbeddings%20%28model%3D%27onlplab%2Falephbert-base%27%2C%20layers%3D%27-1%27%2C%20subtoken_pooling%3D%27first%27%2C%20fine_tune%3DTrue%2C%20use_context%3DTrue%29

# https://github.com/flairNLP/flair/blob/master/resources/docs/embeddings/TRANSFORMER_EMBEDDINGS.md

# ## ver https://github.com/flairNLP/flair/issues/3190

# ### for models for portugues : https://huggingface.co/models?sort=trending&search=portuguese

# # Label dictionary

# In[16]:


from flair.embeddings import TransformerWordEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer


# 2. what label do we want to predict?
label_type = 'ner'
#label_type = 'bio'  # AT ??


# 3. make the label dictionary from the corpus
label_dict = corpus.make_label_dictionary(label_type=label_type, add_unk=False)
# manually add 'O'
label_dict.add_item('O')

if top == 20:
    label_dict.add_item("I_DESIGNAÇÃO_PARA_UMA_ENTIDADE_TERRITORIAL_ADMINISTRATIVA_DE_UM_PAÍS_ESPECÍFICO")
    label_dict.add_item("I_TÁXON")
    
if top == 5:
    label_dict.add_item("I_ANO")
    
#if top == 10:
#    label_dict.add_item("I_ANO")


print(f"LABEL DICT = {label_dict}")


# In[17]:


# new way of define labels Set


# In[18]:


# aux
df1 = pd.read_csv(data_folder+"train.csv",sep="\t",header = None)

df1.columns =["word","ner","line"]
df1.head()


# In[19]:


# aux
df2 = pd.read_csv(data_folder+"dev.csv",sep="\t",header = None)

df2.columns =["word","ner","line"]
df2.head()


# In[20]:


# aux
df3 = pd.read_csv(data_folder+"test.csv",sep="\t",header = None)

df3.columns =["word","ner","line"]
df3.head()


# In[21]:


labels1 = list(df1.ner.unique())
labels1


# In[22]:


labels2 = df2.ner.unique()
labels2


# In[23]:


labels3 = df3.ner.unique()
labels3


# In[24]:


import numpy as np
import math
ner_labels =  [
    x for x in labels1
    if not (
        (isinstance(x, float) and math.isnan(x)) or
        (isinstance(x, str) and x.strip().lower() == "nan")
    )
]

print(f"CUSTOM DICT: {ner_labels}")


# ### use my list as dictionary

# In[25]:


from flair.data import Dictionary

#custom_labels = ['LPerson', 'LLocation', 'LDate']  # your labels as-is
label_dict = Dictionary(add_unk=False)
for label in ner_labels:
    label_dict.add_item(label)

print(label_dict.idx2item)


# In[26]:


# 4. initialize fine-tuneable transformer embeddings WITH document context
# bert-base-uncased

# model='bert-base-multilingual-cased' # de https://github.com/flairNLP/flair/issues/2777
# model= 'bert-base-uncased' # 1st used
# model = "flax-community/portuguese-roberta-base"
# model="bert-base-portuguese-cased" # ver https://huggingface.co/neuralmind/bert-base-portuguese-cased
# model='portuguese-roberta-base'
#model="neuralmind/bert-base-portuguese-cased"
print(f"modelo_base = {modelo_base}")
embeddings = TransformerWordEmbeddings(model=modelo_base,
                                       layers="-1",
                                       subtoken_pooling="first",
                                       fine_tune=True,
                                       use_context=True,
                                       )

# 5. initialize bare-bones sequence tagger (no CRF, no RNN, no reprojection)
tagger = SequenceTagger(hidden_size=128, # was 256,
                        embeddings=embeddings,
                        tag_dictionary=label_dict,
                        tag_type='ner',
                        use_crf=True,
                        use_rnn=False,
                        reproject_embeddings=False,
                        )


# # train it

# #### experiment
# trainer.train(
#     base_path=model_created,
#     learning_rate=5.0e-5,
#     mini_batch_size=4,
#     max_epochs=MAX_EPOCHS,
#     patience=5,                  # Enable early stopping
#     #monitor_train=False,
#     monitor_test=False,
#     #main_evaluation_metric='f1', # Use F1 score on dev set
#     embeddings_storage_mode='none',
#     min_learning_rate = 5.0e-7
# )

# In[46]:


MAX_EPOCHS = 10
print(f"\nFinetuning the model... max = {MAX_EPOCHS}\n")
# 6. initialize trainer
trainer = ModelTrainer(tagger, corpus)

#from flair.trainers import ModelTrainer, TrainingArgs

# 7. run fine-tuning
#training_args = TrainingArgs(
#    output_dir=model_created,
#    max_epochs=MAX_EPOCHS,
#    learning_rate=0.1,
#    mini_batch_size=16,
#    patience=3,             # Stop if no improvement for 3 validations
#    monitor='dev/f1',       # Monitor F1 score on the dev set
#    embeddings_storage_mode='none'
#)

#trainer.fine_tune(training_args)



trainer.fine_tune(model_created,
                  learning_rate=5.0e-6,
                  mini_batch_size=4 , #  2 era valor a 10 junho, was 4,
                  #mini_batch_chunk_size=1,  # remove this parameter to speed up computation if you have a big GPU
                  #max_epochs = 5, # ADDED AT 17 oct 2023
                  max_epochs = MAX_EPOCHS, # mudado em 24 de maio 2025
                  # NEW 13 jun 2025 below
                  )


# In[ ]:





# ##  test it

# In[ ]:


# model_created =f'/home/ematos/phd/pt/NEW_INDOMAIN_NER_FLAIR/data_bert_and_mapper_2025'
#model_created ="/phd/pt/NEW_INDOMAIN_NER_FLAIR/data_bert_and_mapper_2025"


# In[35]:


from flair.data import Sentence
from flair.nn import Classifier


# In[36]:


# make a sentence
#sentence = Sentence('I love Aveiro and Portugal .')
#sentence = Sentence('O palerma do Einstein gostaria de Aveiro e Portugal .')
sentence = Sentence("Paulo Rita trabalhava na Viatura Médica do Hospital de Vila Franca de Xira e na Força Aérea.")
# load the NER tagger
tagger = Classifier.load(model_created+"/final-model.pt")

# run NER over sentence
tagger.predict(sentence)

# print the sentence with all annotations
print(sentence)


# In[37]:


print(corpus.train[5])


# In[38]:


sent = Sentence(corpus.train[12].text)
print(sent)


# In[39]:


tagger.predict(sent)


# In[40]:


print(sent)


# In[41]:


for s in range (10,15):
    sent = Sentence(corpus.train[s].text)
    tagger.predict(sent)
    print(sent)


# ### teste com multiword

# In[42]:


# load the NER tagger
print(f"\nLoading the tagger model...\n")
tagger = Classifier.load(model_created+"/final-model.pt")


# In[43]:


# make a sentence
sentence = Sentence('A Flórida é um estado do sul dos Estados Unidos.')
 
# run NER over sentence
tagger.predict(sentence)

# print the sentence with all annotations
print(sentence)


# ### out of domain

# In[ ]:





# In[44]:


sentence = Sentence("Paulo Rita trabalhava na Viatura Médica do Hospital de Vila Franca de Xira e na Força Aérea.")

# run NER over sentence
tagger.predict(sentence)

# print the sentence with all annotations
print(sentence)


# ### test with all test set

# In[45]:


print(f"\nTesting with Test part of the set..\n")
print(modelo_base_short)
filen = outputdir+"flair-"+modelo_base_short+"-output-test.txt"
print(filen)


# In[ ]:


results=open(filen,"w",encoding="utf8")

for i in range(len(corpus.test)):
    sent = Sentence(corpus.test[i].text)
    #sent
    tagger.predict(sent)
    print(sent.text)
    print(f"\t{sent.labels}")

    results.write(sent.text+"\n")
    results.write(f"\t{sent.labels}\n")

results.close()


# In[ ]:


sent.text


# In[ ]:


sent.labels


# 

# In[ ]:





# In[ ]:


len(corpus.train)


# In[ ]:





# In[ ]:





# In[ ]:




