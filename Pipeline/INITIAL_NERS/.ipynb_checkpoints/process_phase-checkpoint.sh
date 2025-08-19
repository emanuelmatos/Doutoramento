#!/bin/bash

# Ensure script does not stop on errors
set +e

filename="$1.txt"
dirname="$1"

echo "Processing $filename ..........................................................................."

# Copy file to be processed to aux1.txt (file assumed by all NERs)
cp "$filename" aux1.txt

# Preprocess to solve some formatting issues [uses aux1.txt as input, output=input.txt]
echo "Preprocessing input file..."
python preprocess.py
if [ $? -ne 0 ]; then
    echo "Preprocessing failed! Continuing to next steps."
fi

echo "Processing $filename ..........................................................................."

# NER 1 - Allen [out=.allen]
echo "Running NER 1 - Allen..."
if [ ! -f "output.allen" ]; then
    #python allennlp_script_V2_14april2021.py
    python allennlp_script_V2_19dec2024.py $filename
    if [ $? -ne 0 ]; then
        echo "NER 1 - Allen failed! Skipping to next step."
    else
        echo "NER 1 - Allen completed successfully."
    fi
else
    echo "NER 1 - Allen output file already exists. Skipping."
fi


echo "Processing $filename ..........................................................................."

# NER 2 - Linguakit [out=.linguakit]
echo "Running NER 2 - Linguakit..."
if [ ! -f "output.linguakit" ]; then
    #python linguakit_script_14april.py
    python linguakit_script_19dec2024.py $filename
    if [ $? -ne 0 ]; then
        echo "NER 2 - Linguakit failed! Skipping to next step."
    else
        echo "NER 2 - Linguakit completed successfully."
    fi
else
    echo "NER 2 - Linguakit output file already exists. Skipping."
fi

echo "Processing $filename ..........................................................................."

# NER 3 - DBPEDIA [out=.dbpedia]
echo "Running NER 3 - DBPEDIA..."
if [ ! -f "output.dbpedia" ]; then
    #python DBPEDIA_NER_V7_14april2021.py
    python DBPEDIA_NER_V7_19dec2024.py $filename
    if [ $? -ne 0 ]; then
        echo "NER 3 - DBPEDIA failed! Skipping to next step."
    else
        echo "NER 3 - DBPEDIA completed successfully."
    fi
else
    echo "NER 3 - DBPEDIA output file already exists. Skipping."
fi


echo "Processing $filename ..........................................................................."

# Combine NER outputs into a single file
echo "Combining NER outputs..."
python script_join_v4AT_17april2021.py
if [ $? -ne 0 ]; then
    echo "Combining NER outputs failed! Continuing to next steps."
fi

# Decide and create final BIO column
echo "Creating final BIO column..."
python Decision_V3_19april2021.py
if [ $? -ne 0 ]; then
    echo "BIO column creation failed! Continuing to next steps."
fi

# Make graphical output
echo "Generating graphical output..."
python GraphicRepresentationV3-16april2021.py
if [ $? -ne 0 ]; then
    echo "Graphical output generation failed! Continuing to next steps."
fi

# Results for paper (now is Jupyter notebook)
echo "Generating results for paper..."
python ResultPaperSLATE_V1_16april2021.py
if [ $? -ne 0 ]; then
    echo "Result generation failed! Continuing to next steps."
fi

# Create a directory with the same name as the text file and move all output files there
echo "Saving output files to directory: $dirname"
mkdir -p "$dirname"
mv output.* "$dirname/" 2>/dev/null
mv input.txt "$dirname/" 2>/dev/null
mv output-*.* "$dirname/" 2>/dev/null
mv crosstab*.* "$dirname/" 2>/dev/null
mv debug*.* "$dirname/" 2>/dev/null


echo "Processing completed successfully (with potential skipped steps)!$filename .........................................................................."