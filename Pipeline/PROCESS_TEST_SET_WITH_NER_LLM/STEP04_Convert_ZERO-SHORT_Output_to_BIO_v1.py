#!/usr/bin/env python
# coding: utf-8

# # imports

# In[22]:


# === Imports ===
import os
import time
import re
import pandas as pd
import numpy as np
 
 
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# # get info from config file

# In[ ]:





# In[23]:


working_dir = os.getcwd()
print(working_dir)

parts = working_dir.split("/")

if parts[-1].startswith("pipeline"): ### BUG SOLVED MAY 2025
    pipeline = working_dir
else:
    pipeline ="/".join(parts[:-1])
print(pipeline)


# In[24]:


import sys 
base = pipeline
sys.path.append(base)

from config.configsite import *

#print(f"filesdir = {filesdir}") # new AT april 2025
print(f"pipeline = {pipeline}") # new AT april 2025


# In[25]:


# directory with file concatenated_output_sentences....
inputdir=f"{pipeline}{dataset}_PROCESSED/"+resultsdir   
print(f"input dir = {inputdir}") # new AT may 2025


# In[26]:


outputdir=pipeline+"PROCESS_TEST_SET_WITH_NER_LLM/"+resultsdir
print(f"output dir = {outputdir}") # new AT may 2025

os.makedirs(outputdir, exist_ok=True)


# In[27]:


filename_results = f"XML_ZERO-SHOT_sabiazinho3_{dataset}_pt_512_20250615.txt"  # is CSV


# In[28]:


df = pd.read_csv(outputdir+filename_results, sep="\t", on_bad_lines='skip', quoting=3)
print(df.tail())


# In[14]:


result = df.frase_tag
result[:4]


# In[15]:


import re
from xml.etree import ElementTree as ET

def xml_to_bio(xml_text):
    xml_text_wrapped = f"<root>{xml_text}</root>"
    root = ET.fromstring(xml_text_wrapped)

    tokens = []
    bio_tags = []

    def tokenize(text):
        return re.findall(r"\w+|[^\w\s]", text, re.UNICODE)

    def process_node(node, current_tag=None):
        if node.text:
            words = tokenize(node.text)
            for i, word in enumerate(words):
                if current_tag:
                    tag = f"B-{current_tag}" if i == 0 else f"I-{current_tag}"
                else:
                    tag = "O"
                tokens.append(word)
                bio_tags.append(tag)

        for child in node:
            process_node(child, child.tag)
            if child.tail:
                tail_words = tokenize(child.tail)
                for word in tail_words:
                    tokens.append(word)
                    bio_tags.append("O")

    process_node(root)
    return list(zip(tokens, bio_tags))

 


# In[16]:


# Example
input_text = "I met <PERSON>John Doe</PERSON> in <LOCATION>Paris</LOCATION>."
output = xml_to_bio(input_text)
for token, tag in output:
    print(f"{token}\t{tag}")


# In[17]:


fout = open(outputdir+"test_ZERO-SHOT_results_BIO.tsv","w",encoding ="utf8")
fout.write(f"word\ttag\n")

for f in result:
    #print(f"{f}")

    try:
        output = xml_to_bio(f)
    except Exception  as e:
        print(f"Problems - {e}")
  
    for token, tag in output:
        print(f"{token}\t{tag}")

        fout.write(f"{token}\t{tag}\n")

    print()  # empry line

fout.close()


# # look at dataframe

# In[19]:


df = pd.read_csv(outputdir+"test_ZERO-SHOT_results_BIO.tsv",sep="\t", on_bad_lines='skip',  quoting=3)

df[df.tag!="O"]


# In[ ]:




