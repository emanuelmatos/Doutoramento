#!/usr/bin/env python
# coding: utf-8

# In[39]:
import os
import json
import sys
from textblob import TextBlob
from tqdm import tqdm  # Import tqdm for progress tracking
import pandas as pd
#-----------------------------------------------------------------------------------------
from configsite import basedir,inputdir,outputdir

direxit=basedir
filesdir=basedir

#______________________________________________________________________________________


def run_command(command, success_message, error_message):
    """
    Executes a shell command and prints corresponding messages.
    Stops execution if the command fails.
    """
    try:
        subprocess.run(command, shell=True, check=True)
        print(success_message)
    except subprocess.CalledProcessError as e:
        print(f"{error_message}: {e}")
        sys.exit(1)  # Stop execution on failure



# Directory paths
direxit = basedir
filesdir = inputdir

# CSV file containing the list
nome_lista = sys.argv[1]

lista_path = os.path.join(basedir, nome_lista)

# Read the list
try:
    lista = pd.read_csv(lista_path)
except Exception as e:
    print(f"Failed to read the CSV file: {e}")
    sys.exit(1)  # Stop execution if there's an error reading the CSV






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


global cache
cache= {"António Teixeira":"Person"}  # inicialização egocêntrica :)

if os.path.exists("cache.json"):
    cache = json.load( open( "cache.json" ) )



#------------------------------------------------------------------------
def getType(string, tipo):
    global cache

    # Preprocess the input string
    string = string.replace('\r\n', ' ').replace('\n', ' ').replace('"', '').strip()
    string = string.title()  # Optionally capitalize all words

    # Return "NOTFOUND" if the string is too short
    if len(string) < 3:
        return "NOTFOUND"

    # Check cache for the string
    if string in cache:
        #print(f"Cache hit: {string}")
        return cache.get(string)

    # Fetch description from DBpedia
    r = get_description4(string, tipo)
    if len(r["results"]["bindings"]) > 0:
        aux = r["results"]["bindings"][0]['tipo']['value']
        aux2 = basename(normpath(aux))
        sup = getSuperClassV2(aux2)

        # Store result in cache
        cache[string] = f"{aux2};{sup}"
        return f"{aux2};{sup}"
    else:
        return "NOTFOUND"



#-------------------------------------------------------------------
def getTypePT(string, tipo):
    global cache

    # Preprocess the input string
    string = string.replace('\r\n', ' ').replace('\n', ' ').replace('"', '').strip()
    string = string.title()  # Optionally capitalize all words

    # Return "NOTFOUND" if the string is too short
    if len(string) < 3:
        return "NOTFOUND"

    # Check cache for the string
    if string in cache:
        #print(f"Cache hit PT: {string}")
        return cache.get(string)

    # Fetch description from DBpedia (Portuguese)
    r = get_description4PT(string, tipo)
    if len(r["results"]["bindings"]) > 0:
        aux = r["results"]["bindings"][0]['tipo']['value']
        aux2 = basename(normpath(aux))
        sup = getSuperClassV2(aux2)

        # Store result in cache
        cache[string] = f"{aux2};{sup}"
        return f"{aux2};{sup}"
    else:
        return "NOTFOUND"

 


#-------------------------------------------------------------
def get_description4(string, query_type):
    """
    Fetches the description of a string from DBpedia using SPARQL.

    Args:
        string (str): The string to query.
        query_type (str): The type of query to perform (e.g., "rdf:type").

    Returns:
        dict: The result of the SPARQL query in JSON format.
    """
    # Initialize SPARQL endpoint
    sparql = SPARQLWrapper("http://dbpedia.org/sparql")
    sparql.setReturnFormat(JSON)

    # Define SPARQL prefixes and query
    prefix = "PREFIX dbres: <http://dbpedia.org/resource/>\n"
    query_body = (
        f'SELECT ?tipo WHERE {{ '
        f'?page rdfs:label "{string}"@en ; {query_type} ?tipo . '
        f'FILTER strStarts(str(?tipo), "http://dbpedia.org/ontology") }}'
    )
    query = prefix + query_body

    # Set the query
    sparql.setQuery(query)

    # Execute the query and return the result
    try:
        result = sparql.query().convert()
        return result
    except Exception as e:
        print(f"Error executing SPARQL query: {e}")
        return {"results": {"bindings": []}}

 
#-----------------------------------------------------------------------------------------
def get_description4PT(string, query_type):
    """
    Fetches the description of a string in Portuguese from DBpedia using SPARQL.

    Args:
        string (str): The string to query.
        query_type (str): The type of query to perform (e.g., "rdf:type").

    Returns:
        dict: The result of the SPARQL query in JSON format.
    """
    # Initialize SPARQL endpoint
    sparql = SPARQLWrapper("http://dbpedia.org/sparql")
    sparql.setReturnFormat(JSON)

    # Define SPARQL prefixes and query
    prefix = "PREFIX dbres: <http://dbpedia.org/resource/>\n"
    query_body = (
        f'SELECT ?tipo WHERE {{ '
        f'?page rdfs:label "{string}"@pt ; {query_type} ?tipo . '
        f'FILTER strStarts(str(?tipo), "http://dbpedia.org/ontology") }}'
    )
    query = prefix + query_body

    # Set the query
    sparql.setQuery(query)

    # Execute the query and return the result
    try:
        result = sparql.query().convert()
        return result
    except Exception as e:
        print(f"Error executing SPARQL query for {string}: {e}")
        return {"results": {"bindings": []}}

 

#-----------------------------------------------------------------------------------------
#  generalized function that processes word sequences
def seqnwords(df, tokens_list, n):
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
            json.dump(cache, open("cache.json", 'w'))

        # Build n-word sequence
        seq_words = " ".join(tokens_list[pos:pos + n]).strip()

        # Try tagging the sequence in Portuguese, fallback to English
        try:
            tag = getTypePT(seq_words, "rdf:type")
            if tag == "NOTFOUND":
                tag = getType(seq_words, "rdf:type")
        except Exception as e:
            print(f"Error processing sequence: {e}")
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




#-----------------------------------------------------------------------------------------
def super_class(subclass):
    """
    Queries DBpedia to find the superclass of a given subclass.

    Args:
        subclass (str): The subclass to query.

    Returns:
        dict: The result of the SPARQL query in JSON format.
    """
    # Initialize SPARQL endpoint
    sparql = SPARQLWrapper("http://dbpedia.org/sparql")
    sparql.setReturnFormat(JSON)

    # Define SPARQL prefixes and query
    prefix = "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>\n"
    prefix2 = "PREFIX dbo: <http://dbpedia.org/ontology/>\n"
    query_body = (
        f"SELECT ?superclass WHERE {{ "
        f"dbo:{subclass} rdfs:subClassOf ?superclass . "
        f"FILTER strStarts(str(?superclass), 'http://dbpedia.org/ontology') }}"
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



#-----------------------------------------------------------------------------------------
def getSuperClassV2(subclass):
    """
    Recursively retrieves the top-level superclass for a given subclass from DBpedia.

    Args:
        subclass (str): The subclass to start the search.

    Returns:
        str: The top-level superclass name or the most recent valid superclass found.
    """
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
            current_class = basename(normpath(superclass_url))
        else:
            print("NO RESULTS!")
            break

        # Stop if the superclass is a terminal class
        if current_class in {"Agent"}:
            break

        previous_class = current_class

    print(f"Top-level superclass found: {previous_class}")
    return previous_class



#-----------------------------------------------------------------------------------------
def process_file2(filein, fileout, max_n):
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
        json.dump(cache, open("cache.json", 'w'))
        df.to_csv(f"{fileout}_{n}.csv", index=True)

    # Save final results
    df.to_csv(fileout, index=True)
 

#-----------------------------------------------------------------------------------------
def init_from_textfile2(filename, tokens_list):
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

    print(f"Processing file content of type: {type(text)}")

    # Preprocess the text if needed (disabled by default)
    preprocess = False
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

#-------------------------------------------------------------------------
import os
import sys
import json

# Set up directories and filenames #############################################################
# Loop through each line in the DataFrame
for index, row in lista.iterrows():
    filename = row['Filename']
    basename = row['BaseName']
    dir_with_input_txt_files = filesdir
    input_file = inputdir+filename
    print(input_file)
    output_dir = outputdir+basename
    output_file = "output.dbpediaNER"
    dirname = outputdir+basename
    max_n = 7

    


    # Ensure the output directory exists
    ensure_directory_exists(output_dir)

    ###############################################################################################
    # Change to the input directory
    os.chdir(dir_with_input_txt_files)
    
    # Check if the output file already exists
    if os.path.exists(output_file):
        print(f"DBPEDIA NER *** ALERT! File with processing results already exists ({output_file}). "
              f"Delete it if you want to process and rerun.")
        sys.exit("DONE")
    
    # Process the input file
    process_file2(os.path.join(dir_with_input_txt_files, input_file),
                  os.path.join(output_dir, output_file),
                  max_n)
    
    # Save the cache to a file
    with open("cache.json", "w") as cache_file:
        json.dump(cache, cache_file)

    # Create a directory with the same name as the text file and move all output files there
    print(f"Saving output files to directory: {dirname}")
    os.makedirs(dirname, exist_ok=True)

    for pattern in ["output.*", "input.txt", "output-*.*", "crosstab*.*", "debug*.*"]:
        for file in [f for f in os.listdir('.') if os.path.isfile(f) and f.startswith(pattern.split('*')[0])]:
            shutil.move(file, os.path.join(dirname, file))