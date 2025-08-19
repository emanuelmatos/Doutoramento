#!/usr/bin/env python
# coding: utf-8

# In[39]:

import json
import sys

#-----------------------------------------------------------------------------------------
from configsite import filesdir

direxit=filesdir


#______________________________________________________________________________________

# Ensure a parameter (filename) is passed
if len(sys.argv) < 2:
    print("Usage: python process.py <filename>")
    sys.exit("ERROR: Missing parameter")

# Retrieve the parameter (e.g., "Abrolhos")
file_parameter = sys.argv[1]  # This will be "Abrolhos" in the example
print(f"Processing with parameter: {file_parameter}")

# Configurations
from configsite import dataDir

# Input and output file names
filename = sys.argv[1]





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
#----------------------   DBPEDIA ACCESS functions AT, March 2021
def  getType(string, tipo):
    global cache
    
    string=string.replace('\r\n', ' ')
    string=string.replace('\n', ' ')
    #string=string.replace(',', '')   # TODO: think about
    #string=string.replace('.', '')
    string=string.replace('"', '')
    string=string.strip()

    # UNCOMMENT if wanted to capitalize all words
    string=string.title()
    
    #print(string)
    
    #print ("PROCESSING : |"+string+"|") # DEBUG
    if len(string) < 3:
        return "NOTFOUND"
    
    ##  check cache
    if string in cache:
        print("Cache hit: {}".format(string))
        return cache.get(string)
    
    r=get_description4(string,tipo)
    
    
    
    if len(r["results"]["bindings"]) >0:
        #print(r)
        aux=r["results"]["bindings"][0]['tipo']['value']
       
        aux2= basename(normpath(aux))
        sup=getSuperClassV2(aux2)
        cache[string] = aux2+";"+sup #basename(normpath(aux));
    else:
        return("NOTFOUND")
        
        
    
    return aux2+";"+sup   # TODO , mudar para 



#-------------------------------------------------------------------
def  getTypePT(string, tipo):
    global cache
    string=string.replace('\r\n', ' ')
    string=string.replace('\n', ' ')
    #string=string.replace(',', '')
    #string=string.replace('.', '')
    string=string.replace('"', '')
    string=string.strip()
    
    # UNCOMMENT if wanted
    string=string.title()

    #print(string)

    #print ("PROCESSING : |"+string+"|")   # DEBUG
    if len(string) < 3:
        return "NOTFOUND"
    
     ##  check cache
    if string in cache:
        print("Cache hit PT: {}".format(string))
        return cache.get(string)
    
    r=get_description4PT(string,tipo)
  
    if len(r["results"]["bindings"]) >0:
        #print(r)
        aux=r["results"]["bindings"][0]['tipo']['value']
        aux2= basename(normpath(aux))
        #print(aux2)
        #print(type(aux2))
        sup=getSuperClassV2(aux2)
        #print(type(sup))
        cache[string] = aux2+";"+sup #basename(normpath(aux));
    else:
        return("NOTFOUND")
        
        
    
    return aux2+";"+sup   # TODO , mudar para 

 


#-------------------------------------------------------------
def get_description4(string,type):
    #string= "Marcelo Rebelo de Sousa"   # debug

    sparql = SPARQLWrapper("http://dbpedia.org/sparql")
    sparql.setReturnFormat(JSON)

    prefix="PREFIX dbres: <http://dbpedia.org/resource/>\n"
    #print(prefix)
    #aux1="SELECT ?tipo WHERE \{ ?athlete rdfs:label {}@en \}".format(string)
    
    #  versao de QUERY para obter dbo:xx
    aux1='SELECT ?tipo WHERE { ?page rdfs:label "'+string+'"@en ;'+type+' ?tipo . FILTER  strStarts(str(?tipo), "http://dbpedia.org/ontology") }'  #  try to get dbo:* part
    #  versão do QUERY que não filtra (dá muito lixo)
    #aux1='SELECT ?tipo WHERE { ?page rdfs:label "'+string+'"@en ;'+type+' ?tipo .  }'  #  try to get dbo:* part
    # usando schema
    # aux1='SELECT ?tipo WHERE { ?page rdfs:label "'+string+'"@en ;'+type+' ?tipo . FILTER  strStarts(str(?tipo), "http://schema.org/") }'  #  try to get dbo:* part
    
    #print(aux1)
    query=prefix+aux1
    
    #print("QUERY: "+query)  # DEBU
    
    sparql.setQuery(query)  # the previous query as a literal string

    
    aux=sparql.query()
    #print(aux)
    #return "OK"
    #return sparql.query().convert()
    return aux.convert()

 
#-----------------------------------------------------------------------------------------
def get_description4PT(string,type):
    sparql = SPARQLWrapper("http://dbpedia.org/sparql")
    sparql.setReturnFormat(JSON)

    prefix="PREFIX dbres: <http://dbpedia.org/resource/>\n"
    #print(prefix)
    #aux1="SELECT ?tipo WHERE \{ ?athlete rdfs:label {}@en \}".format(string)
    #aux1='SELECT ?tipo WHERE { ?page rdfs:label "'+string+'"@pt ;'+type+' ?tipo . }'  #  try to get dbo:* part
    aux1='SELECT ?tipo WHERE { ?page rdfs:label "'+string+'"@pt ;'+type+' ?tipo . FILTER  strStarts(str(?tipo), "http://dbpedia.org/ontology") }'  #  try to get dbo:* part
    #aux1='SELECT ?tipo WHERE { ?page rdfs:label "'+string+'"@pt ;'+type+' ?tipo . FILTER  strStarts(str(?tipo), "http://schema.org/") }'  #  try to get dbo:* part
    
    #print(aux1)
    query=prefix+aux1
    #print(query)
    sparql.setQuery(query)  # the previous query as a literal string

    return sparql.query().convert()


 

#-----------------------------------------------------------------------------------------
#  generalized function that processes word sequences
def seqnwords(df,tokens_list,n):
    #   seqs de n palavras
    print('\n========== Processing {} word sequences =========='.format(n))
    
    for pos in range(0,len(tokens_list)-n+1):
        percent= pos / len(tokens_list) *100.0
        
        if pos % 100 ==0:
            print(f"{pos} / {len(tokens_list)} = {percent:5.1f} %")

        if pos % 500 == 0:
            print("saving cache")
            json.dump( cache, open( "cache.json", 'w' ) )


        seqnwords=tokens_list[pos].strip()
        for i in range(1,n):
            seqnwords=seqnwords+" "+tokens_list[pos+i].strip() 
    
        #print(seqnwords)

    
        # 1st try Portuguese ...
        try: 
            tag=getTypePT(seqnwords, "rdf:type")
        
        #print(tag)
        #print(type(tag))
        # ... if not found in Portuguese try English
            if tag == "NOTFOUND":
                tag=getType(seqnwords, "rdf:type")

        # TODO: exception handling can be improved to RETRY. for now the idea is not to crash
        except:
            print("Oops!", sys.exc_info()[0], "occurred.")
            tag="NOTFOUND"
    
        #print(seqnwords+" > "+tag)  # DEBUG
        if tag!="NOTFOUND": 
            # tag start of NE
            
            print("TAG >"+tag)
            if n==1:
                df.at[pos,'BIO']="U-"+tag
            else:
                
                df.at[pos,'BIO']="B-"+tag
        
                #  tag following n-2 words
                for i in range(1,n-1):
                    df.at[pos+i,'BIO']="I-"+tag

                # tag last
                df.at[pos+n-1,'BIO']="L-"+tag

        
            print(seqnwords+" > "+tag)
        





#-----------------------------------------------------------------------------------------
def init_from_textfile(filename,tokens_list):
    #---------------------------------  process text 
    # Using readlines() 
    file1 = open(filename, 'r',encoding='utf8') 
    Lines = file1.readlines() 

    lista=[]  
    #tokens_list=[]
    count = 0
    # Strips the newline character 
    for line in Lines: 
        #print("Line{}: {}".format(count, line.strip())) 
        line=line.replace(","," ,")
        line=line.replace("."," .")
    
        tokens=line.split(" ")
        for tok in tokens:
            #x,y=enumerate(tok)
            #print("Token: {}".format(tok))
    
         
            lista.append([count, tok, "O"])

            tokens_list.append(tok)
            count+=1
        
    df= pd.DataFrame(lista, columns=['n', 'Token', 'BIO'])
                        #   index=['a', 'b', 'c', 'd', 'e', 'f'])
    df.set_index("n",inplace=True)
    df.to_csv('output1lista.txt', index=True)
    
    print(df)
    return df





#-----------------------------------------------------------------------------------------
def super_class(subclass):
    
    sparql = SPARQLWrapper("http://dbpedia.org/sparql")
    sparql.setReturnFormat(JSON)

    #prefix="PREFIX dbres: <http://dbpedia.org/resource/>\n"
    prefix="PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>\n"
    prefix2="PREFIX dbo: <http://dbpedia.org/ontology/>\n"

    aux1="select ?superclass where { dbo:"+subclass+ ' rdfs:subClassOf ?superclass . FILTER  strStarts(str(?superclass), "http://dbpedia.org/ontology")  }'
    #.format(subclass)
    
    #aux1='SELECT ?tipo WHERE { ?page rdfs:label "'+string+'"@en ;'+type+' ?tipo . FILTER  strStarts(str(?tipo), "http://dbpedia.org/ontology") }'  #  try to get dbo:* part
    
    #print(aux1)
    query=prefix+prefix2+aux1
    
    #print("QUERY: "+query)  # DEBU
    
    sparql.setQuery(query)  # the previous query as a literal string

    
    aux=sparql.query()
    #aux2=aux.convert()
    #print(aux2)
    #print(type(aux2))

    # {'head': {'link': [], 'vars': ['superclass']}, 'results': {'distinct': False, 'ordered': True, 'bindings': []}}
    #return "OK"
    #return sparql.query().convert()
    return aux.convert()




#-----------------------------------------------------------------------------------------
def getSuperClassV2(str):
    aux2=str
    previous=aux2
        
    while (aux2 != "Thing") &  (aux2 != "Agent"):
  
        print("Looking for "+aux2)

        r=super_class(aux2)
        #display (r)
    
        x=r["results"]["bindings"]
        #print(x)
        if len(r["results"]["bindings"]) >0:
            aux=r["results"]["bindings"][0]['superclass']['value']
            
            aux2=basename(normpath(aux))
            #aux2
            #previous=aux2
        else:
            print("NO RESULTS!")
            failed=True
            #print(aux2)
            break
            
        if aux2 in ['Agent']:  # add more 
            break
        else:
            previous=aux2
            
    print(previous)   
    return previous

   
#-----------------------------------------------------------------------------------------
def process_file (filein, fileout,n):
    global df
    tokens_list=[]
    df=init_from_textfile(filein,tokens_list)

    for n in range(1,6):
        seqnwords(df, tokens_list, n)
    

    df.to_csv(fileout, index=True)   
    
    




#-----------------------------------------------------------------------------------------
def process_file2 (filein, fileout,n):
    global df
    tokens_list=[]
    df=init_from_textfile2(filein,tokens_list)

    for n in range(1,nn+1):
        seqnwords(df, tokens_list, n)
        
        json.dump( cache, open( "cache.json", 'w' ) )
        df.to_csv(fileout+str(n), index=True) 



    df.to_csv(fileout, index=True)   

 

 
import sys  
from importlib import reload
reload(sys)  
sys.getdefaultencoding() # use this for Python3
from textblob import TextBlob
from unidecode import unidecode
 
import re
 

#-----------------------------------------------------------------------------------------
def init_from_textfile2(filename,tokens_list):
    
    url = filename
    file=open(url, encoding="utf-8")
    #file=open(url) 
    t=file.read()
    print(type(t))


    ##  AT 15 april REMOVED after creating preprocess.py
    # some replacements
    use=False
    if use:
        t=t.replace('“', '')   # eliminate "
        t=t.replace('-', ' ')
        t=t.replace('”', '')
        t=re.sub('\.([A-Z0-9])', '. \1', t)   # TODO: fazer isto ao ficheiro de input 
        t=re.sub('\s+', ' ', t)

    #print(t)

    #blob = TextBlob(unidecode(t))
    blob = TextBlob(t)

    lista=[]  
    #tokens_list=[]
    count = 0
    
    frases=blob.sentences
    for frase in frases:
        #print(frase)
        palavras = frase.words
        for p in palavras:
            print(p)
            lista.append([count, p, "O"])

            tokens_list.append(p)
            count+=1
        
        lista.append([count, ".", "O"])
        tokens_list.append(".")
        count+=1
        
        use=False     # use or not <EOS> after period.  Doesn't seem necessary
        if use:
            lista.append([count, "<EOS>", "O"])
            tokens_list.append("<EOS>")
            count+=1
        
        print(".")
        print("\n")
        
         
        
    df= pd.DataFrame(lista, columns=['n', 'Token', 'BIO'])
                        #   index=['a', 'b', 'c', 'd', 'e', 'f'])
    df.set_index("n",inplace=True)
    df.to_csv('output1lista.txt', index=True)
    
    print(df)
    return df



#-----------------------------------------------------------------------------------------
from configsite import filesdir

#dirWithInputTxtFiles=r"C:\Users\AJST\Dropbox\EMANUEL_PhD\NewTourismMiniCorpusByEmanuel"
#dirWithInputTxtFiles="C:\\Users\\AJST\\Dropbox\\EMANUEL_PhD\\NER_COMPLETE_SYSTEM_V1\\"
dirWithInputTxtFiles=filesdir

inputFile="input.txt"
#inputFile="inputshort.txt"  # only for debug


outDir=dirWithInputTxtFiles
outFile="output.dbpediaNER"
nn=7

os.chdir(dirWithInputTxtFiles)
if os.path.exists(outFile):  
    print(f"DBPEDIA NER *** ALERT! File with processing results already exists ({outFile}). Delete it if you want to process and rerun.")
    sys.exit("DONE")


#process_file2(r"C:\Users\AJST\Dropbox\EMANUEL_PhD\NewTourismMiniCorpusByEmanuel\textosEmanuelClean2ANSI.txt","outEmanuelClean1_nerv5.txt",5)
process_file2(dirWithInputTxtFiles+inputFile, outDir+outFile,nn)
 



#import json
json.dump( cache, open( "cache.json", 'w' ) )

