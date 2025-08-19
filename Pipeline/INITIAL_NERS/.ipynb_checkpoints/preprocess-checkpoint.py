import os
import re

import sys  
from importlib import reload
reload(sys)  
sys.getdefaultencoding() # use this for Python3
from textblob import TextBlob
from unidecode import unidecode

from configsite import filesdir

#filesdir="C:\\Users\\AJST\\Dropbox\\EMANUEL_PhD\\NER_COMPLETE_SYSTEM_V1\\"

inputfile="aux1.txt"
outfile="input.txt"

os.chdir(filesdir)

file=open(inputfile, encoding="utf-8")
#file=open(url) 
t=file.read()
print(type(t))


##  TODO: REMOVE CLOCK after creating preprocess.py
# some replacements
t=t.replace('“', '')   # eliminate "
t=t.replace('-', ' ')
t=t.replace('”', '')

#t=re.sub('\.([A-Z0-9])', '. \1', t)   # TODO: fazer isto ao ficheiro de input 
t=re.sub('\s+', ' ', t)

blob = TextBlob(t)

#  write to file
fileout=open(outfile,"w", encoding="utf-8")

    
frases=blob.sentences
for frase in frases:
    fileout.write(str(frase)+"\n")


fileout.close()