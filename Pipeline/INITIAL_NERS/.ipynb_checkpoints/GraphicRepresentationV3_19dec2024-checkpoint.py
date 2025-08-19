#!/usr/bin/env python
# coding: utf-8

# Import necessary libraries
import requests
import re
import spacy
from spacy.tokens import Doc
from spacy import displacy
from pathlib import Path
import pandas as pd

# Load SpaCy English model
nlp = spacy.load("en_core_web_sm")

# Define helper function to preprocess BIO annotations
def preprocess_bio_tags(entities):
    """Convert BIO tags to a consistent format."""
    entities = [item.replace('U-', 'B-') for item in entities]
    entities = [item.replace('L-', 'I-') for item in entities]
    return entities

# Load CSV file
filename = "output.decision"
df = pd.read_csv(filename, delimiter=',')

# Extract words and entities from the dataframe
words = list(df['WORD'])
entities_list = {
    "BIO": list(df['BIO']),
    "BIO1": list(df['BIO1']),
    "BIO2": list(df['BIO2']),
    "BIO3": list(df['BIO3']),
    "ENTITY": list(df['ENTITY']),
}

# Define colors and display options for visualization
colors = {
    "NP00G00": "linear-gradient(90deg, #aa9cfc, #fc9ce7)", 
    "NP00V00": "linear-gradient(90deg, #119cfc, #119ce7)"
}
options = {"ents": ["NP00G00", "NP00V00"], "colors": colors}

# Process each entity type and generate visualizations
for key, entities in entities_list.items():
    print(f"Processing {key}...")
    
    # Preprocess BIO tags
    res = preprocess_bio_tags(entities)
    
    # Ensure words and entities have the same length
    if len(res) != len(words):
        print(f"Length mismatch for {key}: words={len(words)}, entities={len(res)}")
        continue
    
    # Create a SpaCy Doc object
    doc = Doc(nlp.vocab, words=words, ents=res)
    print(f"Sample text for {key}: {doc[:10].text}")
    
    # Generate visualization as an SVG
    svg = displacy.render(doc, style="ent", jupyter=False, options=options)
    output_path = Path(f"./output-{key.lower()}.html")
    output_path.open("w", encoding="utf-8").write(svg)
    print(f"Visualization for {key} saved to {output_path}")

print("Processing complete.")