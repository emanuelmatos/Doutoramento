#!/usr/bin/env python
# coding: utf-8

# # create the conditions to run

# In[1]:


#cd /home/ematos/ateixeira/pipelineDBPEDIALOCAL


# In[2]:


#!pwd


# In[3]:


#!pip install SPARQLWrapper
#!pip install nltk

import nltk
# un comment if neeed
#nltk.download('all')


# In[4]:


#!/usr/bin/env python
# coding: utf-8

# In[39]:
import os
import json
import sys
from textblob import TextBlob
from tqdm import tqdm  # Import tqdm for progress tracking
import pandas as pd


# In[5]:


from os.path import basename


# # setup dirs

# In[6]:


import os

working_dir = os.getcwd()
print(working_dir)

parts = working_dir.split("/")
if parts[-1].startswith("pipeline"):
    pipeline = working_dir
else:
    pipeline ="/".join(parts[:-1])
print(pipeline)


# In[7]:


#base = "/home/ematos/ateixeira/pipeline/"
#base = "/home/ematos/phd/pt/"
base = pipeline
sys.path.append(base)

#-----------------------------------------------------------------------------------------
from config.configsite import * #basedir,inputdir

direxit=basedir
filesdir=basedir

#______________________________________________________________________________________





# Directory paths
direxit = basedir
filesdir = inputdir

outputdir=pipeline+"INITIAL_NERS/"+resultsdir+"/"
print(f"output dir = {outputdir}") # new AT april 2025

#outputdir = '/home/ematos/ateixeira/pipeline/INITIAL_NERS/OUTPUT_RESULTS/arte_guerra/'


# In[8]:


# Add the base directory to the system path
#base = "/home/ematos/Documentos/phd/NER/process1/process_script/"
sys.path.append(pipeline)

# Import directories from configsite
#from configsite import filesdir, linguakitdir, inputdir, outputdir,basedir

#______________________________________________________________________________________


# # funções

# ### run_command

# In[9]:


def run_command(command, success_message, error_message,debug=False):
    """
    Executes a shell command and prints corresponding messages.
    Stops execution if the command fails.
    """
    if debug:
        print(f"EXECUTING: {command}")
    
    try:
        subprocess.run(command, shell=True, check=True)
        print(success_message)
    except subprocess.CalledProcessError as e:
        print(f"ERROR in run_comman() : {error_message}: {e}")
        sys.exit(1)  # Stop execution on failure


# # ler CSV com lista de files do dataset

# In[10]:


# Nome da lista de arquivos a processar
nome_lista = f'{pipeline}/{dataset}_PROCESSED/files_list.csv'


# Read the list
try:
    lista = pd.read_csv(nome_lista)
except Exception as e:
    print(f"Failed to read the CSV file: {e}")
    sys.exit(1)  # Stop execution if there's an error reading the CSV


# # initialize SPQRQL wrapper and cache

# In[11]:


""" My first try at NER based in DBPEDIA in Python """

from pandas.errors import DataError
#from bloom_filter import BloomFilter
import numpy
import pandas as pd
import json

from SPARQLWrapper import SPARQLWrapper, JSON
from os.path import normpath, basename
import os

# In[40]:


# # cache

# In[12]:


global cache
cache= {"António Teixeira":"Person"}  # inicialização egocêntrica :)

cache_dir = pipeline+"INITIAL_NERS/"
cache_file = cache_dir+"cache.json"

if os.path.exists(cache_dir+"cache.json"):
    print(f"DBPEDIA Cache read from {cache_file}")
    cache = json.load( open( cache_file ) )


# # more functions

# ## getType()
r = {'head': {'link': [], 'vars': ['tipo']}, 'results': {'distinct': False, 'ordered': True, 'bindings': [{'tipo': {'type': 'uri', 'value': 'http://dbpedia.org/ontology/Work'}}, {'tipo': {'type': 'uri', 'value': 'http://dbpedia.org/ontology/Film'}}]}}

if len(r["results"]["bindings"]) > 0:
    print(">0")
    aux = r["results"]["bindings"][0]['tipo']['value']
    print(aux)

    partes = aux.split("/")
    print(partes)

    aux2 = partes[-1]
    print(aux2)

    #sup = getSuperClassV2(aux2)

 
# In[13]:


#------------------------------------------------------------------------
def getType(string, tipo, debug = False):
    global cache

    if debug:
        print(f"--- getType()")

    # Preprocess the input string
    string = string.replace('\r\n', ' ').replace('\n', ' ').replace('"', '').strip()
    string = string.title()  # Optionally capitalize all words

    if debug:
        print(f"getType ({string}, {tipo} )")

    # Return "NOTFOUND" if the string is too short
    
    if len(string) < 3:
        return "NOTFOUND"

    # Check cache for the string
    if string in cache:
        print(f"Cache hit in getType: {string}")
        return cache.get(string)

    # Fetch description from DBpedia
    r = get_description4(string, tipo)

    if debug:
        print(f"\n\nr = {r}\n")

    # EXEMPLO: {'head': {'link': [], 'vars': ['tipo']}, 'results': {'distinct': False, 'ordered': True, 'bindings': [{'tipo': {'type': 'uri', 'value': 'http://dbpedia.org/ontology/Work'}}, {'tipo': {'type': 'uri', 'value': 'http://dbpedia.org/ontology/Film'}}]}}
    
    if len(r["results"]["bindings"]) > 0:
        aux = r["results"]["bindings"][0]['tipo']['value']

        #print(aux)

        partes = aux.split("/")
        #print(partes)

        aux2 = partes[-1]
        #print(aux2)
        #aux2 = basename(normpath(aux))
        sup = getSuperClassV2(aux2)

        # Store result in cache
        cache[string] = f"{aux2};{sup}"
        return f"{aux2};{sup}"
    else:
        return "NOTFOUND"


# ### getTypeLocal

# In[14]:


#------------------------------------------------------------------------
def getTypeLocal(string, tipo, debug = False):
    global cache
    if debug:
        print(f"--- getTypeLocal()")
    # Preprocess the input string
    string = string.replace('\r\n', ' ').replace('\n', ' ').replace('"', '').strip()
    string = string.title()  # Optionally capitalize all words

    if debug:
        print(f"getTypeLocal( {string}, {tipo} )")

    # Return "NOTFOUND" if the string is too short
    
    if len(string) < 3:
        return "NOTFOUND"

    # Check cache for the string
    if string in cache:
        print(f"Cache hit in getType: {string}")
        return cache.get(string)

    # Fetch description from DBpedia
    r = get_description4_local(string, tipo)   # NEW JULY 2025

    if debug:
        print(f"\n\nr = {r}\n")

    # EXEMPLO: {'head': {'link': [], 'vars': ['tipo']}, 'results': {'distinct': False, 'ordered': True, 'bindings': [{'tipo': {'type': 'uri', 'value': 'http://dbpedia.org/ontology/Work'}}, {'tipo': {'type': 'uri', 'value': 'http://dbpedia.org/ontology/Film'}}]}}
    
    if len(r["results"]["bindings"]) > 0:
        aux = r["results"]["bindings"][0]['tipo']['value']

        #print(aux)

        partes = aux.split("/")
        #print(partes)

        aux2 = partes[-1]
        #print(aux2)
        #aux2 = basename(normpath(aux))
        sup = getSuperClassV2Local(aux2) ## NEW JULY 2025

        # Store result in cache
        cache[string] = f"{aux2};{sup}"
        return f"{aux2};{sup}"
    else:
        return "NOTFOUND"


# ## getTypePT

# In[15]:


#-------------------------------------------------------------------
def getTypePT(string, tipo, debug = False):
    global cache
    if debug:
        print(f"--- getTypePT()")
    # Preprocess the input string
    string = string.replace('\r\n', ' ').replace('\n', ' ').replace('"', '').strip()
    string = string.title()  # Optionally capitalize all words

    if debug:
        print(f"getTyepPt( {string}, {tipo} )")

    # Return "NOTFOUND" if the string is too short
    if len(string) < 3:
        return "NOTFOUND"

    # Check cache for the string
    if string in cache:
        if debug:
            print(f"Cache hit PT: {string}")
        return cache.get(string)

    # Fetch description from DBpedia (Portuguese)
    r = get_description4PT(string, tipo)
    
    if debug:
        print(f"\n\nr = {r}\n")
    
    if len(r["results"]["bindings"]) > 0:
        aux = r["results"]["bindings"][0]['tipo']['value']
        #print(aux)

        partes = aux.split("/")
        #print(partes)

        aux2 = partes[-1]
        #print(aux2)
        #aux2 = basename(normpath(aux))
        sup = getSuperClassV2(aux2)

        # Store result in cache
        cache[string] = f"{aux2};{sup}"
        return f"{aux2};{sup}"
    else:
        return "NOTFOUND"

 


# ### getTypePTLocal

# In[16]:


def getTypePTLocal(string, tipo, debug = False):
    global cache
    
    if debug:
        print(f"--- getTypePTLocal()")

    # Preprocess the input string
    string = string.replace('\r\n', ' ').replace('\n', ' ').replace('"', '').strip()
    string = string.title()  # Optionally capitalize all words

    if debug:
        print(f"getTyepPTLocal ( {string}, {tipo} )")

    # Return "NOTFOUND" if the string is too short
    if len(string) < 3:
        return "NOTFOUND"

    # Check cache for the string
    if string in cache:
        if debug:
            print(f"Cache hit PT: {string}")
        return cache.get(string)

    # Fetch description from DBpedia (Portuguese)
    r = get_description4PT_local(string, tipo)   #NEW JULY 2025  <<--- LOCAL
    
    if debug:
        print(f"\n\nr = {r}\n")
    
    if len(r["results"]["bindings"]) > 0:
        aux = r["results"]["bindings"][0]['tipo']['value']
        #print(aux)

        partes = aux.split("/")
        #print(partes)

        aux2 = partes[-1]
        #print(aux2)
        #aux2 = basename(normpath(aux))
        sup = getSuperClassV2Local(aux2)  # <<--- LOCAL

        # Store result in cache
        cache[string] = f"{aux2};{sup}"
        return f"{aux2};{sup}"
    else:
        return "NOTFOUND"


# ## get_description4()

# In[17]:


#-------------------------------------------------------------
def get_description4(string, query_type, debug = False):
    """
    Fetches the description of a string from DBpedia using SPARQL.

    Args:
        string (str): The string to query.
        query_type (str): The type of query to perform (e.g., "rdf:type").

    Returns:
        dict: The result of the SPARQL query in JSON format.
    """
    if debug:
        print(f"--- get_description4()")
    
    query = ""
    if debug:
        print(f"+++ get_description4 ( {string}, {query_type} )")
        
    # Initialize SPARQL endpoint
    sparql = SPARQLWrapper("https://dbpedia.org/sparql")

 

    
    # https://dbpedia.org/sparql
    sparql.setReturnFormat(JSON)

    # Define SPARQL prefixes and query
    prefix = "PREFIX dbres: <http://dbpedia.org/resource/>\n"
    query_body = (
        'SELECT ?tipo WHERE {  { '   
        f'?page rdfs:label "{string}"@en ; {query_type} ?tipo . '
        'FILTER strStarts(str(?tipo), "http://dbpedia.org/ontology") }}'
    )
    query = prefix + query_body

    if debug:
        print(f"QUERY = {query_body}")

    # Set the query
    sparql.setQuery(query)

    # Execute the query and return the result
    try:
        result = sparql.query().convert()
        if debug:
            print(f"SPARQL Result = {result}")
       
    except Exception as e:
        print(f"Error executing SPARQL query: {e}")
        result = {"results": {"bindings": []}}

    #if debug:
    #        print(f"get_description4 returning {result}")
    return result

 


# ### get_description4_local

# In[18]:


def get_description4_local(string, query_type, debug = False, debug2=False):
    """
    Fetches the description of a string from DBpedia using SPARQL.

    Args:
        string (str): The string to query.
        query_type (str): The type of query to perform (e.g., "rdf:type").

    Returns:
        dict: The result of the SPARQL query in JSON format.
    """
    if debug2:
        print(f"--- get_description4_local()")
    
    query = ""
    if debug:
        print(f"+++ get_description4_local ( {string}, {query_type} )")
        
    # Initialize SPARQL endpoint
    #sparql = SPARQLWrapper("https://dbpedia.org/sparql")

    # NEW 9 JULY 2025
    sparql = SPARQLWrapper("http://hlt.ieeta.pt:8890/sparql")

    
    # https://dbpedia.org/sparql
    sparql.setReturnFormat(JSON)

    # Define SPARQL prefixes and query
    prefix = "PREFIX dbres: <http://dbpedia.org/resource/>\n"
    query_body = (
        'SELECT ?tipo WHERE { GRAPH <http://localhost:8890/DBPEDIA> { '   # updated 9 jULY 2025
        f'?page rdfs:label "{string}"@en ; {query_type} ?tipo . '
        'FILTER strStarts(str(?tipo), "http://dbpedia.org/ontology") }}'
    )
    query = prefix + query_body

    if debug:
        print(f"QUERY = {query_body}")

    # Set the query
    sparql.setQuery(query)

    # Execute the query and return the result
    try:
        result = sparql.query().convert()
        if debug2:
            print(f"SPARQL Result = {result}")
       
    except Exception as e:
        print(f"Error executing SPARQL query LOCALLY: {e}")
        result = {"results": {"bindings": []}}

    #if debug:
    #        print(f"get_description4 returning {result}")
    return result


# ## get_description4PT

# In[19]:


#-----------------------------------------------------------------------------------------
def get_description4PT(string, query_type, debug = False):
    """
    Fetches the description of a string in Portuguese from DBpedia using SPARQL.

    Args:
        string (str): The string to query.
        query_type (str): The type of query to perform (e.g., "rdf:type").

    Returns:
        dict: The result of the SPARQL query in JSON format.
    """
    if debug:
        print(f"--- get_description4PT()")
    query = ""

    if debug:
        print(f"QUERY PT = {query}")
    
    # Initialize SPARQL endpoint
    sparql = SPARQLWrapper("https://dbpedia.org/sparql")
    
    sparql.setReturnFormat(JSON)

    # Define SPARQL prefixes and query
    prefix = "PREFIX dbres: <http://dbpedia.org/resource/>\n"
    query_body = (
        'SELECT ?tipo WHERE { { '
        f'?page rdfs:label "{string}"@pt ; {query_type} ?tipo . '
        'FILTER strStarts(str(?tipo), "http://dbpedia.org/ontology") }}'
    )
    query = prefix + query_body

    # Set the query
    sparql.setQuery(query_body)

    # Execute the query and return the result
    try:
        result = sparql.query().convert()
        if debug:
            print(f"SPARQL Result = {result}")
        
        
    except Exception as e:
        print(f"Error executing SPARQL query for {string}: {e}")
        result = {"results": {"bindings": []}}

    # if debug:
    #    print(f"get_description4PT() returning {result}")

    return result


# ### get_description4PT_local

# In[20]:


def get_description4PT_local(string, query_type, debug = False,debug2=False):
    """
    Fetches the description of a string in Portuguese from DBpedia using SPARQL.

    Args:
        string (str): The string to query.
        query_type (str): The type of query to perform (e.g., "rdf:type").

    Returns:
        dict: The result of the SPARQL query in JSON format.
    """
    if debug2:
        print(f"--- get_description4PT_local()")
    query = ""

    if debug:
        print(f"QUERY PT = {query}")
    
    # Initialize SPARQL endpoint
    #sparql = SPARQLWrapper("https://dbpedia.org/sparql")
    sparql = SPARQLWrapper("http://hlt.ieeta.pt:8890/sparql")
    sparql.setReturnFormat(JSON)

    # Define SPARQL prefixes and query
    prefix = "PREFIX dbres: <http://dbpedia.org/resource/>\n"
    query_body = (
        'SELECT ?tipo WHERE {GRAPH <http://localhost:8890/DBPEDIA> { '
        f'?page rdfs:label "{string}"@pt ; {query_type} ?tipo . '
        'FILTER strStarts(str(?tipo), "http://dbpedia.org/ontology") }}'
    )
    query = prefix + query_body

    # Set the query
    sparql.setQuery(query_body)

    # Execute the query and return the result
    try:
        result = sparql.query().convert()
        if debug2:
            print(f"SPARQL Result PT = {result}")
        
        
    except Exception as e:
        print(f"Error executing SPARQL query LOCALLY for {string}: {e}")
        result = {"results": {"bindings": []}}

    # if debug:
    #    print(f"get_description4PT() returning {result}")

    return result


# ## seqnwords()

# In[21]:


#-----------------------------------------------------------------------------------------
#  generalized function that processes word sequences
def seqnwords(df, tokens_list, n, debug = False, debug2 = True):
    """
    Processes sequences of n words from tokens_list and tags them using DBpedia.

    Args:
        df (DataFrame): DataFrame containing tokens and their BIO tags.
        tokens_list (list): List of tokens to process.
        n (int): Number of words in each sequence.

    Returns:
        None: Updates the DataFrame in place.
    """
    print(f'\n========== Processing {n}-word sequences ==========')

    total_tokens = len(tokens_list)
    for pos in range(total_tokens - n + 1):
        # Progress reporting every 100 steps
        if pos % 100 == 0:
            percent = (pos / total_tokens) * 100.0
            print(f"{pos} / {total_tokens} = {percent:.1f}% completed")

        # Save cache periodically
        if pos % 500 == 0:
            print("Saving cache...")
            json.dump(cache, open(cache_file, 'w'))

        # Build n-word sequence
        seq_words = " ".join(tokens_list[pos:pos + n]).strip()

        # Try tagging the sequence in Portuguese, fallback to English
        try:
            if debug2:
                print(f"seq_words = {seq_words}")

            # ALGORITMO: tentar com  PT, se falhar tentar ENG
            tag = getTypePT(seq_words, "rdf:type")
            if tag == "NOTFOUND":
                tag = getType(seq_words, "rdf:type")

            if debug:
                print(f"TAG = {tag}")
        except Exception as e:
            print(f"Error processing sequence: {e}, {seq_words}")
            tag = "NOTFOUND"

        # If a tag is found, assign BIO tags
        if tag != "NOTFOUND":
            #print(f"TAG > {tag}")
            if n == 1:
                df.at[pos, 'BIO'] = "U-" + tag
            else:
                df.at[pos, 'BIO'] = "B-" + tag
                # Tag intermediate tokens
                for i in range(1, n - 1):
                    df.at[pos + i, 'BIO'] = "I-" + tag
                # Tag the last token
                df.at[pos + n - 1, 'BIO'] = "L-" + tag

            #print(f"{seq_words} > {tag}")


# ### seqnwords_local

# In[22]:


def seqnwords_local(df, tokens_list, n, debug = False, debug2 = False):
    """
    Processes sequences of n words from tokens_list and tags them using DBpedia.

    Args:
        df (DataFrame): DataFrame containing tokens and their BIO tags.
        tokens_list (list): List of tokens to process.
        n (int): Number of words in each sequence.

    Returns:
        None: Updates the DataFrame in place.
    """
    print(f'\n========== Processing {n}-word sequences (DBPEDIA LOCAL) ==========')

    total_tokens = len(tokens_list)
    for pos in range(total_tokens - n + 1):
        # Progress reporting every 100 steps
        if pos % 1000 == 0:
            percent = (pos / total_tokens) * 100.0
            print(f"LOCAL | {pos} / {total_tokens} = {percent:.1f}% completed")

        # Save cache periodically
        if pos % 1000 == 0:
            print("Saving cache...")
            json.dump(cache, open(cache_file, 'w'))

        # Build n-word sequence
        seq_words = " ".join(tokens_list[pos:pos + n]).strip()

        # Try tagging the sequence in Portuguese, fallback to English
        try:
            if debug2:
                print(f"seq_words = {seq_words}")

            # ALGORITMO: tentar com  PT, se falhar tentar ENG
            tag = getTypePTLocal(seq_words, "rdf:type")
            if tag == "NOTFOUND":
                tag = getTypeLocal(seq_words, "rdf:type")

            if debug:
                print(f"TAG = {tag}")
        except Exception as e:
            print(f"Error processing sequence: {e}, {seq_words}")
            tag = "NOTFOUND"

        # If a tag is found, assign BIO tags
        if tag != "NOTFOUND":
            #print(f"TAG > {tag}")
            if n == 1:
                df.at[pos, 'BIO'] = "U-" + tag
            else:
                df.at[pos, 'BIO'] = "B-" + tag
                # Tag intermediate tokens
                for i in range(1, n - 1):
                    df.at[pos + i, 'BIO'] = "I-" + tag
                # Tag the last token
                df.at[pos + n - 1, 'BIO'] = "L-" + tag

            #print(f"{seq_words} > {tag}")


# ### seqnwords_eval_local

# In[23]:


def seqnwords_eval_local(df, tokens_list, n, debug = False, debug2 = True):
    """  ...
    """
    print(f'\n========== Processing {n}-word sequences ==========')

    total_tokens = len(tokens_list)
    for pos in range(total_tokens - n + 1):
        # Progress reporting every 100 steps
        if pos % 100 == 0:
            percent = (pos / total_tokens) * 100.0
            print(f"{pos} / {total_tokens} = {percent:.1f}% completed")

        # Save cache periodically
        #if pos % 500 == 0:
        #    print("Saving cache...")
        #    json.dump(cache, open(cache_file, 'w'))

        # Build n-word sequence
        seq_words = " ".join(tokens_list[pos:pos + n]).strip()

        # PARTE 1 - using REMOTE DBPEDIA
        try:
            if debug:
                print(f"+ REMOTE DBPEDIA")
            # ALGORITMO: tentar com  PT, se falhar tentar ENG
            tag1 = getTypePT(seq_words, "rdf:type")
            if tag1 == "NOTFOUND":
                tag1 = getType(seq_words, "rdf:type")

        except Exception as e:
            print(f"Error processing sequence: {e}, {seq_words}")
            tag = "NOTFOUND"

        # PARTE 2 - using LOCAL DBPEDIA
        try:
            if debug:
                print(f"+ LOCAL DBPEDIA") 
            # ALGORITMO: tentar com  PT, se falhar tentar ENG
            taglocal = getTypePTLocal(seq_words, "rdf:type")
            if taglocal == "NOTFOUND":
                taglocal = getTypeLocal(seq_words, "rdf:type")

        except Exception as e:
            print(f"Error processing sequence: {e}, {seq_words}")
            taglocal = "NOTFOUND"

        # If a tag is found..
        if (tag1 != "NOTFOUND") or (taglocal != "NOTFOUND"):
            print(f"\n{seq_words} > REMOTE={tag1}\tLOC={taglocal}")
            if tag1 != taglocal:
                print(f"\t\t\t!!!! ATENCAO !!!!")


# ## init_from_text_file()

# In[24]:


#-----------------------------------------------------------------------------------------
def init_from_textfile(filename, tokens_list):
    """
    Initializes a DataFrame from a text file and populates a token list.

    Args:
        filename (str): Path to the input text file.
        tokens_list (list): List to append tokens extracted from the text.

    Returns:
        DataFrame: A DataFrame containing tokens and their initial BIO tags.
    """
    # Read the file content
    with open(inputdir+filename, 'r', encoding='utf8') as file:
        lines = file.readlines()

    # Initialize lists for DataFrame construction
    data = []
    count = 0

    # Process each line
    for line in lines:
        line = line.replace(",", " ,").replace(".", " .")
        tokens = line.split()
        for token in tokens:
            data.append([count, token, "O"])
            tokens_list.append(token)
            count += 1

    # Create a DataFrame
    df = pd.DataFrame(data, columns=['n', 'Token', 'BIO'])
    df.set_index("n", inplace=True)

    # Save to a file for inspection
    df.to_csv('output1lista.txt', index=True)

    #print(df)
    return df


# ## super_class()

# In[25]:


#-----------------------------------------------------------------------------------------
def super_class(subclass, debug =False):
    """
    Queries DBpedia to find the superclass of a given subclass.

    Args:
        subclass (str): The subclass to query.

    Returns:
        dict: The result of the SPARQL query in JSON format.
    """
    # Initialize SPARQL endpoint
    if debug:
        print("--- super_class()")
    
    sparql = SPARQLWrapper("http://dbpedia.org/sparql")
    
    sparql.setReturnFormat(JSON)

    # Define SPARQL prefixes and query
    prefix = "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>\n"
    prefix2 = "PREFIX dbo: <http://dbpedia.org/ontology/>\n"
    query_body = (
        "SELECT ?superclass WHERE {{ "
        f"dbo:{subclass} rdfs:subClassOf ?superclass . "
        "FILTER strStarts(str(?superclass), 'http://dbpedia.org/ontology') }}"
    )
    query = prefix + prefix2 + query_body

    # Set and execute the query
    try:
        sparql.setQuery(query)
        result = sparql.query().convert()
        return result
    except Exception as e:
        print(f"Error executing SPARQL query for subclass {subclass}: {e}")
        return {"results": {"bindings": []}}


# ### super_class_local

# In[26]:


def super_class_local(subclass, debug =False):
    """
    Queries DBpedia to find the superclass of a given subclass.

    Args:
        subclass (str): The subclass to query.

    Returns:
        dict: The result of the SPARQL query in JSON format.
    """
    if debug:
        print("--- super_class_local()")
    
    # Initialize SPARQL endpoint
    sparql = SPARQLWrapper("http://hlt.ieeta.pt:8890/sparql")
 

    
    #sparql = SPARQLWrapper("http://dbpedia.org/sparql")
    
    sparql.setReturnFormat(JSON)

    # Define SPARQL prefixes and query
    prefix = "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>\n"
    prefix2 = "PREFIX dbo: <http://dbpedia.org/ontology/>\n"
    query_body = (
        "SELECT ?superclass WHERE {GRAPH <http://localhost:8890/DBPEDIA>{ "
        f"dbo:{subclass} rdfs:subClassOf ?superclass . "
        "FILTER strStarts(str(?superclass), 'http://dbpedia.org/ontology') }}"
    )
    query = prefix + prefix2 + query_body

    # Set and execute the query
    try:
        sparql.setQuery(query)
        result = sparql.query().convert()
        return result
    except Exception as e:
        print(f"Error executing SPARQL query for subclass {subclass}: {e}")
        return {"results": {"bindings": []}}


# ## getSuperClassV2()

# In[27]:


#-----------------------------------------------------------------------------------------
def getSuperClassV2(subclass):
    """
    Recursively retrieves the top-level superclass for a given subclass from DBpedia.

    Args:
        subclass (str): The subclass to start the search.

    Returns:
        str: The top-level superclass name or the most recent valid superclass found.
    """
    print(f"--- getSuperClassV2() -- Looking for superclass of: {subclass}")
    
    current_class = subclass
    previous_class = current_class

    while current_class not in {"Thing", "Agent"}:
        print(f"Looking for superclass of: {current_class}")

        # Query DBpedia for the superclass
        result = super_class(current_class)
        bindings = result.get("results", {}).get("bindings", [])

        if bindings:
            # Extract the superclass value
            superclass_url = bindings[0]['superclass']['value']
            #current_class = basename(normpath(superclass_url))
            current_class = superclass_url.split("/")[-1]   # POROBLEM SOLVED 1 MAY 2025, deve ser da passagem para Linux
            
        else:
            print("NO RESULTS!")
            break

        # Stop if the superclass is a terminal class
        if current_class in {"Agent"}:
            break

        previous_class = current_class

    print(f"Top-level superclass found: {previous_class}")
    return previous_class


# ### getSuperClassV2Local

# In[28]:


def getSuperClassV2Local(subclass):
    """
    Recursively retrieves the top-level superclass for a given subclass from DBpedia.

    Args:
        subclass (str): The subclass to start the search.

    Returns:
        str: The top-level superclass name or the most recent valid superclass found.
    """
    print(f"--- getSuperClassV2Local() -- Looking for superclass of: {subclass}")
    
    current_class = subclass
    previous_class = current_class

    while current_class not in {"Thing", "Agent"}:
        print(f"Looking for superclass of: {current_class}")

        # Query DBpedia for the superclass
        result = super_class_local(current_class)
        bindings = result.get("results", {}).get("bindings", [])

        if bindings:
            # Extract the superclass value
            superclass_url = bindings[0]['superclass']['value']
            #current_class = basename(normpath(superclass_url))
            current_class = superclass_url.split("/")[-1]   # POROBLEM SOLVED 1 MAY 2025, deve ser da passagem para Linux
            
        else:
            print("NO RESULTS!")
            break

        # Stop if the superclass is a terminal class
        if current_class in {"Agent"}:
            break

        previous_class = current_class

    print(f"Top-level superclass found: {previous_class}")
    return previous_class


# ## process_file2()

# In[29]:


#-----------------------------------------------------------------------------------------
def process_file2(filein, fileout, max_n, debug = False):
    """
    Processes a text file to identify n-word sequences and assign BIO tags.

    Args:
        filein (str): Path to the input file.
        fileout (str): Path to the output file prefix.
        max_n (int): Maximum length of n-word sequences to process.

    Returns:
        None: Writes results to files and updates the cache.
    """
    global df
    tokens_list = []

    # Initialize the DataFrame from the input file
    df = init_from_textfile2(filein, tokens_list)

    # Process sequences for 1 to max_n words
   
    for n in tqdm(range(1, max_n + 1)):
        print(f"Processing {n}-word sequences...")
        seqnwords(df, tokens_list, n)

        # Save intermediate results and cache
        json.dump(cache, open(cache_file, 'w'))
        df.to_csv(f"{fileout}_{n}.csv", index=True)

    # Save final results
    df.to_csv(fileout, index=True)
 


# ### process_file2_local

# In[30]:


def process_file2_local(filein, fileout, min_n, max_n, debug = False):
    """
    Processes a text file to identify n-word sequences and assign BIO tags.

    Args:
        filein (str): Path to the input file.
        fileout (str): Path to the output file prefix.
        max_n (int): Maximum length of n-word sequences to process.

    Returns:
        None: Writes results to files and updates the cache.
    """
    global df
    tokens_list = []

    # Initialize the DataFrame from the input file
    df = init_from_textfile2(filein, tokens_list)

    # Process sequences for 1 to max_n words
   
    for n in tqdm(range(min_n, max_n + 1)):
        print(f"Processing {n}-word sequences...")
        seqnwords_local(df, tokens_list, n)

        # Save intermediate results and cache
        json.dump(cache, open(cache_file, 'w'))
        df.to_csv(f"{fileout}_{n}.csv", index=True)

    # Save final results
    df.to_csv(fileout, index=True)


# ## process_file_eval_local_dbpedia()

# In[31]:


def process_file_eval_local_dbpedia(filein, fileout, min_n,max_n, debug = False):
    """
    Processes a text file to identify n-word sequences and assign BIO tags.

    Args:
        filein (str): Path to the input file.
        fileout (str): Path to the output file prefix.
        max_n (int): Maximum length of n-word sequences to process.

    Returns:
        None: Writes results to files and updates the cache.
    """
    global df
    tokens_list = []

    # Initialize the DataFrame from the input file
    df = init_from_textfile2(filein, tokens_list)

    # Process sequences for 1 to max_n words
   
    for n in tqdm(range(min_n, max_n + 1)):
        print(f"Processing {n}-word sequences...")
        seqnwords_eval_local(df, tokens_list, n)

        # Save intermediate results and cache
        #json.dump(cache, open(cache_file, 'w'))
        df.to_csv(f"{fileout}_{n}.csv", index=True)

    # Save final results
    df.to_csv(fileout, index=True)


# In[32]:


#-----------------------------------------------------------------------------------------
import re

def init_from_textfile2(filename, tokens_list, preprocess = True):
    """
    Initializes a DataFrame from a text file and populates a token list.

    Args:
        filename (str): Path to the input text file.
        tokens_list (list): List to append tokens extracted from the text.

    Returns:
        DataFrame: A DataFrame containing tokens and their initial BIO tags.
    """
    # Read the file content
    with open(filename, encoding="utf-8") as file:
        text = file.read()

    # print(f"Processing file content of type: {type(text)}")

    # Preprocess the text if needed (disabled by default)
    #preprocess = False
    if preprocess:
        text = text.replace('“', '').replace('”', '').replace('-', ' ')
        text = re.sub(r'\.([A-Z0-9])', r'. \1', text)  # Ensure proper spacing after periods
        text = re.sub(r'\s+', ' ', text)

    # Use TextBlob for sentence and word tokenization
    blob = TextBlob(text)

    data = []
    count = 0

    # Process each sentence
    for sentence in blob.sentences:
        for word in sentence.words:
            #print(word)
            data.append([count, word, "O"])
            tokens_list.append(word)
            count += 1

        # Append period token
        data.append([count, ".", "O"])
        tokens_list.append(".")
        count += 1

    # Create a DataFrame from the processed data
    df = pd.DataFrame(data, columns=['n', 'Token', 'BIO'])
    df.set_index("n", inplace=True)

    # Save the DataFrame to a file for inspection
    df.to_csv('output1lista.txt', index=True)

    #print(df)
    return df
#-----------------------------------------------------------------------------------------
def ensure_directory_exists(directory):
    #""
    #Ensures that a given directory exists. If it doesn't, create it.
    #"""
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
            print(f"Created directory: {directory}")
        except Exception as e:
            print(f"Failed to create directory {directory}: {e}")
            sys.exit(1)


# In[33]:


#!python -m textblob.download_corpora


# # Avaliar resultados usando versão local e web da DBPEDIA

# In[34]:


def avaliar(min_n = 1,max_n = 7):
    #max_n = 7
    
    file_process = nome_lista
    lista = pd.read_csv(file_process)
        
     
    # Loop through each line in the DataFrame
    for index, row in lista.iterrows():
        filename = row[0]
        basename = str(row[1])
        dir_with_input_txt_files = filesdir
        
        input_file = inputdir+filename
        
        print(f"Processing file = {input_file}, index={index}")
        
        output_dir = outputdir+basename
        output_file = "avaliacao.dbpediaNER"
        dirname = outputdir+basename
       
    
        fin =os.path.join(dir_with_input_txt_files, input_file)
        fout = os.path.join(output_dir, output_file)
    
    
        # Ensure the output directory exists
        ensure_directory_exists(output_dir)
    
        ###############################################################################################
        # Change to the input directory
        #os.chdir(dir_with_input_txt_files)
        
        # Check if the output file already exists
        print(f"Out file = {fout}")
    
        
        if os.path.exists(fout):
            print(f"DBPEDIA NER *** ALERT! File with processing results already exists ({fout}). "
                  f"Delete it if you want to process and rerun.")
            continue
            #sys.exit("DONE")
        
        # Process the input file
    
        print(f"PROCESS FILE2 : {fin} --> {fout}")
        
        process_file_eval_local_dbpedia(fin, fout, min_n, max_n)
        
        # Save the cache to a file
        cache_dir = pipeline+"INITIAL_NERS/"
        #cache_file = cache_dir+"cache.json"
    
        #with open(cache_file, "w") as cfile:
        #    json.dump(cache, cfile)


# # execute  eval DBPEDIA LOCAL (vs remote)

# In[35]:


#avaliar(1,10)


# # process

# In[36]:


def process(max_n = 7):
    #max_n = 7
    
    file_process = nome_lista
    lista = pd.read_csv(file_process)
        
        # Set up directories and filenames #############################################################
        # Loop through each line in the DataFrame
    for index, row in lista.iterrows():
        filename = row[0]
        basename = str(row[1])
        dir_with_input_txt_files = filesdir
        
        input_file = inputdir+filename
        
        print(f"Processing file = {input_file}, index={index}")
        
        output_dir = outputdir+basename
        output_file = "output.dbpediaNER"
        dirname = outputdir+basename
       
    
        fin =os.path.join(dir_with_input_txt_files, input_file)
        fout = os.path.join(output_dir, output_file)
    
    
        # Ensure the output directory exists
        ensure_directory_exists(output_dir)
    
        ###############################################################################################
        # Change to the input directory
        #os.chdir(dir_with_input_txt_files)
        
        # Check if the output file already exists
        print(f"Out file = {fout}")
    
        
        if os.path.exists(fout):
            print(f"DBPEDIA NER *** ALERT! File with processing results already exists ({fout}). "
                  f"Delete it if you want to process and rerun.")
            continue
            #sys.exit("DONE")
        
        # Process the input file
    
        print(f"PROCESS FILE2 : {fin} --> {fout}")
        
        process_file2(fin, fout, max_n)
        
        # Save the cache to a file
        cache_dir = pipeline+"INITIAL_NERS/"
        cache_file = cache_dir+"cache.json"
    
        with open(cache_file, "w") as cfile:
            json.dump(cache, cfile)
    
        # Create a directory with the same name as the text file and move all output files there
        # print(f"Saving output files to directory: {dirname}")
        # os.makedirs(dirname, exist_ok=True)
    
        # for pattern in ["output.*", "input.txt", "output-*.*", "crosstab*.*", "debug*.*"]:
        #    for file in [f for f in os.listdir('.') if os.path.isfile(f) and f.startswith(pattern.split('*')[0])]:
        #        shutil.move(file, os.path.join(dirname, file))


# ### process_local

# In[40]:


def process_local(min_n =1, max_n = 7, keep_if_exists = False):
    #max_n = 7
    
    file_process = nome_lista
    lista = pd.read_csv(file_process)
        
        # Set up directories and filenames #############################################################
        # Loop through each line in the DataFrame
    for index, row in lista.iterrows():
        filename = row[0]
        basename = str(row[1])
        dir_with_input_txt_files = filesdir
        
        input_file = inputdir+filename
        
        print(f"Processing file = {input_file}, index={index}")
        
        output_dir = outputdir+basename
        output_file = "output.dbpediaNER"
        dirname = outputdir+basename
       
    
        fin =os.path.join(dir_with_input_txt_files, input_file)
        fout = os.path.join(output_dir, output_file)
    
    
        # Ensure the output directory exists
        ensure_directory_exists(output_dir)
    
        ###############################################################################################
        # Change to the input directory
        #os.chdir(dir_with_input_txt_files)
        
        # Check if the output file already exists
        print(f"Out file = {fout}")
    
        
        if keep_if_exists  and os.path.exists(fout):
            print(f"DBPEDIA NER *** ALERT! File with processing results already exists ({fout}). "
                  f"Delete it if you want to process and rerun.")
            continue
            #sys.exit("DONE")
        
        # Process the input file
    
        print(f"PROCESS FILE2 : {fin} --> {fout}")
        
        process_file2_local(fin, fout, min_n, max_n)
        
        # Save the cache to a file
        cache_dir = pipeline+"INITIAL_NERS/"
        cache_file = cache_dir+"cache.json"
    
        with open(cache_file, "w") as cfile:
            json.dump(cache, cfile)
    
        # Create a directory with the same name as the text file and move all output files there
        # print(f"Saving output files to directory: {dirname}")
        # os.makedirs(dirname, exist_ok=True)
    
        # for pattern in ["output.*", "input.txt", "output-*.*", "crosstab*.*", "debug*.*"]:
        #    for file in [f for f in os.listdir('.') if os.path.isfile(f) and f.startswith(pattern.split('*')[0])]:
        #        shutil.move(file, os.path.join(dirname, file))


# # execute process com DBPEDIA local

# In[41]:


process_local(1,7)


# In[ ]:


# process(2)


# 
# 
# # The End!

# In[ ]:




