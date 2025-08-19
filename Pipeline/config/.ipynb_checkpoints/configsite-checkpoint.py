##  base directory for all the pipeline (where the global scrip will run)
##  AT, april 2025

# ONGOING: usar var base to simplify vars

#pipeline="/media/ematos/12988235988216FF/Users/EmanuelMatos/phd/pipeline_city/"
pipeline="/home/ematos/phd/pipeline_city/"

resultsdir = "OUTPUT_RESULTS/"  # will be used in all main steps, as a subdir

# ==================  DATASET SECTION ======================

##  directory for the Dataset
#dataset = "DATASET"
dataset = "DATASET_CITY"

inputdir = pipeline+dataset+"/"
processed = pipeline+dataset+"_PROCESSED/"

# ==================  INITIAL NERS SECTION ======================

##  directory
#basedir="/home/ematos/Documentos/phd/NER/process1/"
basedir=pipeline+"INITIAL_NERS/"

## Linguakit dir
#linguakitdir=pipeline_basedir+INITIAL_NERS/Linguakit/"

##  directory for complete system (whe the files must be)
#outputdir="/home/ematos/Documentos/phd/NER/OUTPUT_RESULTS/"
#outputdir="/home/ematos/ateixeira/pipeline/INITIAL_NERS/OUTPUT_RESULTS/"

#filesdir=basedir
datagraph = 'CITY'