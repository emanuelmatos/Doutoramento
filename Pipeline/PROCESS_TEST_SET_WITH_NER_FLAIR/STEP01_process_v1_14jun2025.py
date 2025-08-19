#!/usr/bin/env python
# coding: utf-8

# In[45]:


import pandas as pd
import random

 


# In[46]:


import flair
print(flair.__version__)


# # To save as BIO
# 

# In[47]:


import re
def processLabel(label):
    #print(label)
    #print(f"label type = {type(label)}")

    
    if len(label)>0:
        if isinstance(label, list):
            res = label[0].value
            #print(f"{type(s)}, {s}")
            
        #match = re.search(r"/'([^']+)'", s)
        #if match:
        #    res = match.group(1)
        
        
    else:
        res = "O"
        
    return res # (label,res)

def toBIO(sent):
    for token in sent:
        print(f"{token.text}\t{processLabel(token.labels)}")


# In[48]:


def toBIOfile(sent, file):
    for token in sent:
        file.write(f"{token.text}\t{processLabel(token.labels)}\n")


# ### config experiment

# In[49]:


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


# In[50]:


import sys 
base = pipeline
sys.path.append(base)

from config.configsite import *

#print(f"filesdir = {filesdir}") # new AT april 2025
print(f"pipeline = {pipeline}") # new AT april 2025


# In[51]:


#inputdir=pipeline+"JOIN_MAPPER_REBEL/"+resultsdir
inputdir=pipeline+"NEW_INDOMAIN_NER_FLAIR/"+resultsdir   # AT 24 de maio 2025

print(f"input dir = {inputdir}") # new AT may 2025


# In[52]:


# model dir
 
modeldir=inputdir+"dataset_bert_and_mapper/"

print(f"model dir = {modeldir}") # new AT may 2025


# In[53]:


outputdir=pipeline+"PROCESS_TEST_SET_WITH_NER_FLAIR/"+resultsdir
print(f"output dir = {outputdir}") # new AT may 2025

os.makedirs(outputdir, exist_ok=True)


# # ???

# In[54]:


tag_column = "ner2"  # 1 class only

dataset_dir = inputdir+"dataset_bert_and_mapper/"
print(f"input dataset dir = {dataset_dir}") # new AT may 2025
 


#  # Model file

# In[55]:


model_created = modeldir+"final-model.pt"
print(f"model file to use = {model_created}") 


# In[56]:


import os

# 2025-06-14 13:40:01.963462: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. 
# You may see slightly different numerical results due to floating-point round-off errors from different computation orders. 
# To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.

# Set an environment variable
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


# In[57]:


import flair, torch

device = None
if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('mps')


# In[58]:


print(device)


# #  load the dataset (to be changed)

# In[59]:


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

# 
# # define columns
# columns = {0: 'text', 1: 'ner', 2:'nline'}
# 
# # this is the folder in which train, test and dev files reside
# data_folder = dataset_dir
# 
# # init a corpus using column format, data folder and the names of the train, dev and test files
# corpus: Corpus = ColumnCorpus(data_folder, columns,
#                               train_file='train.csv',
#                               test_file='test.csv',
#                               dev_file='dev.csv')
# 

# You can also access a sentence and check out annotations. Lets assume that the training split is read from the example above, then executing these commands

# print(corpus.train[0].to_tagged_string('ner'))

# print(corpus.train[10].to_tagged_string('nline'))

# # Create Corpus from sentence list  [NEW! 14 de junho]

# In[60]:


from flair.data import Sentence, Corpus, Dataset

# Custom Dataset from a list of Sentence objects
class PlainTextDataset(Dataset):
    def __init__(self, sentences):
        self.sentences = sentences

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return self.sentences[idx]


# ### Read your file (one sentence per line)

# In[61]:


# 
dir_test_sentences = f"{pipeline}{dataset}_PROCESSED/{resultsdir}"
#filename_test_sentences = "concatenated_output_sentences_v2.csv"  # FAKED. is the train. 

filename_test_sentences = "concatenated_test_sentences_v2.csv"  # THe correct SET, the TEST SET 


print(f"teste sentences dir = {dir_test_sentences}")
print(f"file with test sentences = {filename_test_sentences}")


# In[62]:


# Read your file (one sentence per line)

file_path = dir_test_sentences+filename_test_sentences
#with open(file_path, 'r', encoding='utf-8') as f:
#    lines = f.readlines()


# In[63]:


df = pd.read_csv(file_path, sep = ",", header=0)
print(df.head())


# In[64]:


lines1 = list(df.text)
print(lines1[:3])


# In[65]:


#lines = [line.split('\t')[1:][0] for line in lines1]
lines = lines1
#print(lines[:3])


# In[66]:


# Convert lines into flair Sentence objects
sentences = [Sentence(line.strip()) for line in lines if line.strip()]
print(sentences[:3])


# In[67]:


# Split into train/test/dev sets (you can change ratios)
train_split = int(0.999 * len(sentences))  # TODO: passar a 100 %
dev_split = int(0.9995 * len(sentences))

train_dataset = PlainTextDataset(sentences[:train_split])
dev_dataset = PlainTextDataset(sentences[train_split:dev_split])
test_dataset = PlainTextDataset(sentences[dev_split:])

# Create the Corpus
#corpus = Corpus(train=train_dataset, dev=dev_dataset, test=test_dataset)
#corpus = Corpus(train=test_dataset, dev=dev_dataset, test=train_dataset)  # NOTE: test is all the set

# V2
corpus = Corpus(train=sentences, dev=sentences, test=sentences)  # NOTE: test is all the set



# Check the corpus
print(corpus)


# In[71]:


print(corpus.test[0])
print(corpus.test[1])


# ## process 

# In[72]:


#from flair.datasets import CONLL_03
from flair.embeddings import WordEmbeddings, FlairEmbeddings, StackedEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer


# ## code below is from https://flairnlp.github.io/docs/tutorial-training/how-to-train-sequence-tagger

# In[73]:


tagger=None
torch.cuda.empty_cache() 


# ##  ver para mais informação
# https://yulianudelman.medium.com/named-entity-recognition-with-flair-4d627b18f5b7#:~:text=from%20flair.embeddings%20import%20TransformerWordEmbeddings%20embeddings%20%3D,TransformerWordEmbeddings%20%28model%3D%27onlplab%2Falephbert-base%27%2C%20layers%3D%27-1%27%2C%20subtoken_pooling%3D%27first%27%2C%20fine_tune%3DTrue%2C%20use_context%3DTrue%29

# https://github.com/flairNLP/flair/blob/master/resources/docs/embeddings/TRANSFORMER_EMBEDDINGS.md

# ## ver https://github.com/flairNLP/flair/issues/3190

# ### for models for portugues : https://huggingface.co/models?sort=trending&search=portuguese

# In[74]:


# new way of define labels Set


# ##  test it

# In[75]:


# model_created =f'/home/ematos/phd/pt/NEW_INDOMAIN_NER_FLAIR/data_bert_and_mapper_2025'
#model_created ="/phd/pt/NEW_INDOMAIN_NER_FLAIR/data_bert_and_mapper_2025"


# In[76]:


from flair.data import Sentence
#from flair.nn import Classifier

from flair.models import SequenceTagger


# In[77]:


# make a sentence
#sentence = Sentence('I love Aveiro and Portugal .')
#sentence = Sentence('O palerma do Einstein gostaria de Aveiro e Portugal .')
sentence = Sentence("Paulo Rita trabalhava na Viatura Médica do Hospital de Vila Franca de Xira e na Força Aérea.")
# load the NER tagger
#tagger = Classifier.load(model_created)


# OURS  +++++
tagger = SequenceTagger.load(model_created)

#tagger = SequenceTagger.load("flair/ner-portuguese-large")  # or your custom model
#tagger = SequenceTagger.load("ner")  # or your custom model

# run NER over sentence
tagger.predict(sentence)

# print the sentence with all annotations
print(sentence)


# In[78]:


print(corpus.test[5])


# In[79]:


sent = Sentence(corpus.test[12].text)
print(sent)


# In[80]:


tagger.predict(sent)


# In[81]:


print(sent)


# In[ ]:





# In[82]:


for s in range (10,15):
    sent = Sentence(corpus.test[s].text)
    tagger.predict(sent)
    print(sent)


# ### teste com multiword

# In[83]:


# make a sentence
sentence = Sentence('A Flórida é um estado do sul dos Estados Unidos.')
 
# run NER over sentence
tagger.predict(sentence)

# print the sentence with all annotations
print(sentence)


# ### out of domain

# In[84]:


sentence = Sentence("Paulo Rita trabalhava na Viatura Médica do Hospital de Vila Franca de Xira e na Força Aérea.")

# run NER over sentence
tagger.predict(sentence)

# print the sentence with all annotations
print(sentence)


# # test with all test set

# In[85]:


print(f"\nTesting with Test part of the set..\n")

modelo_base_short = "ROBERTA"
#print(modelo_base_short)

filen = outputdir+"flair-"+modelo_base_short+"-output-test.txt"
print(f"Saving results of NER FLAIR on TEst set to {filen}")


# In[88]:


from tqdm import tqdm


results=open(filen,"w",encoding="utf8")

f = open(outputdir+"output_bio.csv", "w", encoding="utf-8") 
f.write("word\ttag\n")

for i in tqdm(range(len(corpus.test))):   
#for i in range(len(corpus.test)):
    sent = Sentence(corpus.test[i].text)
    #sent
    tagger.predict(sent)

    if (i % 300 ==0) or (i<=5):
        print(f"{i+1:4d}:{sent.text}")
        print(f"\t{sent.labels}")

    results.write(sent.text+"\n")
    results.write(f"\t{sent.labels}\n")


    toBIOfile(sent,f)
    # parte BIO
    #for token in sent:
    #    word = token.text
    #    tag = token.get_tag('ner').value
    #    f.write(f"{word}\t{tag}\n")
    #f.write("\n")

results.close()

f.close()


# # test functions fro BIO output

# In[43]:


debug = False

if debug:
    sent = sentences[13]
    
    tagger.predict(sent)
    
    print(sent.text)
    
    print(sent.labels)


# In[44]:


if debug:
    toBIO(sent)


# In[ ]:





# from flair.data import Label
# 
# label = Label("positive", 0.95)
# label_str = label.value  # This will give you "positive"
# print(label_str)
# 

# In[ ]:





# In[ ]:




