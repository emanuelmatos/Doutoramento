#!/usr/bin/env python
# coding: utf-8

# In[4]:


import os
import glob
import pandas as pd
import re
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


# In[24]:


base = pipeline
sys.path.append(base)


from config.configsite import * #basedir,inputdir
 

input_path_train=pipeline+"REBEL_ENTITY2CLASSES/"+resultsdir+"CLASSES/TRAIN/"
print(f"input path = {input_path_train}") # new AT may 2025

input_path_test=pipeline+"REBEL_ENTITY2CLASSES/"+resultsdir+"CLASSES/TEST/"
print(f"input path test set = {input_path_test}") # new AT may 2025

input_path_dev=pipeline+"REBEL_ENTITY2CLASSES/"+resultsdir+"CLASSES/TEST/"
print(f"input path dev set = {input_path_dev}") # new AT may 2025


output_dir=pipeline+"REBEL_ENTITY2CLASSES/"+resultsdir
print(f"output_dir = {output_dir}") 


# # TODO: fazer para TEST set

# In[8]:


# Define the input path and pattern
#input_path = "/home/ematos/phd/pt/REBEL_ENTITY2CLASSES/CLASSES/TRAIN"
#input_path = 

pattern = os.path.join(input_path_train, "results_wiki_train_pt_*.txt")

# Find all matching files
files = glob.glob(pattern)

# List to store DataFrames
df_list = []

# Regex to extract the * part from filename
filename_pattern = re.compile(r"results_wiki_train_pt_(.+)\.txt")

# Read and accumulate content from each file, skipping the first line
for file_path in files:
    try:
        # Extract just the relevant part from the filename
        filename = os.path.basename(file_path)
        match = filename_pattern.match(filename)
        file_id = match.group(1) if match else "unknown"

        # Read the file, skipping the header
        df = pd.read_csv(
            file_path,
            sep='\t',
            names=["line", "mapper", "base", "result"],
            skiprows=1
        )

        # Add the extracted part as a new column
        df["file_id"] = file_id

        df_list.append(df)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")

# Concatenate all DataFrames
final_df = pd.concat(df_list, ignore_index=True)
print(final_df)


# In[15]:


# Save to CSV
output_csv_path = os.path.join(input_path_train, "output_train_final.csv")
final_df.to_csv(output_csv_path, index=False, encoding='utf-8')

print(f"Saved {len(final_df)} rows to {output_csv_path}")


# # do the same for TEST set

# In[13]:


# Define the input path and pattern
#input_path = "/home/ematos/phd/pt/REBEL_ENTITY2CLASSES/RESULTS/TEST"

pattern = os.path.join(input_path_test, "results_wiki_train_pt_*.txt")

# Find all matching files
files = glob.glob(pattern)
#print(files)
if len(files)> 0:

    # List to store DataFrames
    df_list = []
    
    # Regex to extract the * part from filename
    filename_pattern = re.compile(r"results_wiki_train_pt_(.+)\.txt")
    
    # Read and accumulate content from each file, skipping the first line
    for file_path in files:
        try:
            # Extract just the relevant part from the filename
            filename = os.path.basename(file_path)
            match = filename_pattern.match(filename)
            file_id = match.group(1) if match else "unknown"
    
            # Read the file, skipping the header
            df = pd.read_csv(
                file_path,
                sep='\t',
                names=["line", "mapper", "base", "result"],
                skiprows=1
            )
    
            # Add the extracted part as a new column
            df["file_id"] = file_id
    
            df_list.append(df)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
    
    # Concatenate all DataFrames
    final_df = pd.concat(df_list, ignore_index=True)
    
    # Save to CSV
    output_csv_path = os.path.join(input_path, "output_test_final.csv")
    final_df.to_csv(output_csv_path, index=False, encoding='utf-8')
    
    print(f"Saved {len(final_df)} rows to {output_csv_path}")
else:
    print(f"No files found for TEST set")


# In[16]:


# Define the input path and pattern
#input_path = "/home/ematos/phd/pt/REBEL_ENTITY2CLASSES/RESULTS/DEV"
pattern = os.path.join(input_path_dev, "results_wiki_train_pt_*.txt")

# Find all matching files
files = glob.glob(pattern)

if len(files)> 0:

    # List to store DataFrames
    df_list = []
    
    # Regex to extract the * part from filename
    filename_pattern = re.compile(r"results_wiki_train_pt_(.+)\.txt")
    
    # Read and accumulate content from each file, skipping the first line
    for file_path in files:
        try:
            # Extract just the relevant part from the filename
            filename = os.path.basename(file_path)
            match = filename_pattern.match(filename)
            file_id = match.group(1) if match else "unknown"
    
            # Read the file, skipping the header
            df = pd.read_csv(
                file_path,
                sep='\t',
                names=["line", "mapper", "base", "result"],
                skiprows=1
            )
    
            # Add the extracted part as a new column
            df["file_id"] = file_id
    
            df_list.append(df)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
    
    # Concatenate all DataFrames
    final_df = pd.concat(df_list, ignore_index=True)
    
    # Save to CSV
    output_csv_path = os.path.join(input_path, "output_dev_final.csv")
    final_df.to_csv(output_csv_path, index=False, encoding='utf-8')
    
    print(f"Saved {len(final_df)} rows to {output_csv_path}")
else:
    print(f"No files found for DEV set")


# In[22]:


import pandas as pd
aux = os.path.join(input_path_train, "output_train_final.csv")
train = pd.read_csv(aux)
df = train

aux = os.path.join(input_path_test, "output_test_final.csv")
try:
    test = pd.read_csv(aux)
    df = pd.concat([df,test])
except Exception as e:
    print(f"WARN! {e}")

aux = os.path.join(input_path_dev, "output_dev_final.csv")
try:
    dev = pd.read_csv(aux)
    df = pd.concat([df,dev])
except Exception as e:
    print(f"WARN! {e}")

print(df.head())


# In[23]:


#sel = [train,test,dev]
#df = pd.concat(sel)


# In[26]:


filename = output_dir+'results_rebel.csv'
print(f"Saving to {filename}")
df.to_csv(filename, index = None, sep = '\t')


# In[ ]:




