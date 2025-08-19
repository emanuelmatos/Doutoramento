#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NER Data Processing Script
Created for robust and efficient handling of NER outputs with alignment and consensus building.
"""

import os
import pandas as pd
from collections import Counter
#-----------------------------------------------------------------------------------------
from configsite import filesdir

direxit=filesdir
#______________________________________________________________________________________


# Set the working directory and input file path
#txtdir = '/home/ematos/phd/NER_COMPLETE_SYSTEM_V1/'
os.chdir(direxit)

# Load the input DataFrame
filename = "output.join"
df = pd.read_csv(filename, delimiter=',')

# Optionally drop the "BIO3" column (uncomment if needed)
# df.drop(columns=["BIO3"], inplace=True)

# Display the first few rows for verification
print("Initial DataFrame:")
print(df.head())

# Initialize a new DataFrame for additional processing
df2 = pd.DataFrame({
    "TAGS": ["O"] * len(df),        # Aggregated tags
    "TYPES": ["0"] * len(df),       # Entity types
    "NTAGS": [0] * len(df),         # Number of non-"O" tags
    "ENTITY": ["O"] * len(df),      # Entity labels (BIO format)
    "WTA": ["O"] * len(df),         # Winner Takes All label
    "CONSENSUS": ["O"] * len(df),   # Reserved for future consensus logic
    "BIO": ["O"] * len(df)          # Final BIO labels
})

# Step 3: Map Linguakit entity codes to meaningful labels
entity_mapping = {
    "NP00G00": "LOC",
    "NP00SP0": "PER",
    "NP00V00": "MISC",
    "NP00O00": "ORG"
}
# Replace entity codes in BIO2 column
df["BIO2"] = df["BIO2"].replace(entity_mapping, regex=True)

# Standardize BIO formats in BIO2 and BIO1 columns
df["BIO2"] = df["BIO2"].replace({"U-": "B-"}, regex=True)
df["BIO1"] = df["BIO1"].replace({"U-": "B-"}, regex=True)

# Step 4: Combine tags from BIO1, BIO2, and optionally BIO3
df2["TAGS"] = df["BIO1"] + ";" + df["BIO2"] + ";" + df.get("BIO3", "O") + ";"

# Extract types by removing BIO prefixes
df2["TYPES"] = df2["TAGS"].replace({"[UBIL]-": ""}, regex=True)

# Step 5: Calculate NTAGS (number of non-"O" tags)
df2["NTAGS"] = df2["TAGS"].apply(lambda x: 3 - x.count("O;"))

# Display rows with at least 2 tags for verification
print("Rows with NTAGS >= 2:")
print(df2[df2["NTAGS"] >= 2])

# Step 6: Assign ENTITY column based on NTAGS
df2["ENTITY"] = df2.apply(lambda row: "B-ENT" if row["NTAGS"] > 1 else "O", axis=1)

# Adjust consecutive "B-ENT" to "I-ENT" for sequential entities
for i in range(1, len(df2)):
    if df2.loc[i, "NTAGS"] > 1 and df2.loc[i - 1, "NTAGS"] > 1:
        df2.loc[i, "ENTITY"] = "I-ENT"

# Display adjusted entities
print("Adjusted ENTITY column:")
print(df2[df2["ENTITY"] == "I-ENT"])

# Step 7: Implement Winner Takes All (WTA) logic
def wta(tag_str, threshold=2):
    """
    Determines the most common tag based on a threshold.
    """
    tags = tag_str.split(";")
    cnt = Counter(tags)
    most_common, count = cnt.most_common(1)[0]

    # Return "O" if no tag meets the threshold
    if count < threshold:
        return "O"
    return f"B-{most_common}" if most_common != "O" else "O"

# Apply WTA logic to TYPES column
df2["WTA"] = df2["TYPES"].apply(lambda x: wta(x, threshold=2))

# Adjust consecutive WTA labels to I-ENT format
for i in range(1, len(df2)):
    if df2.loc[i, "NTAGS"] > 1 and df2.loc[i - 1, "NTAGS"] > 1:
        df2.loc[i, "WTA"] = df2.loc[i, "WTA"].replace("B-", "I-")

# Display rows with non-"O" WTA labels
print("Rows with non-'O' WTA labels:")
print(df2[df2["WTA"] != "O"])

# Step 8: Merge results into the original DataFrame
df["BIO"] = df2["WTA"]
df["WTA"] = df2["WTA"]
df["ENTITY"] = df2["ENTITY"]

# Display the final merged DataFrame
print("Final DataFrame:")
print(df.head(14))

# Step 9: Save the results to files
df.to_csv("output.decision", index=False)  # Save with header
df.to_csv("output.conll", sep=' ', index=False, header=False)  # Save in CoNLL format

# Export a subset of rows to a LaTeX table
filtered_rows = df[(df["ENTITY"] != "O") | (df["WTA"] != "O")][150:160]
with open("mytable.tex", "w") as tf:
    tf.write(filtered_rows.to_latex(index=False))

print("Processing complete. Results saved.")