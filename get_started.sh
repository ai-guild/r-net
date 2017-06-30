#!/usr/bin/env bash
# Downloads raw data into ./download
# and saves preprocessed data into ./data
# Get directory containing this script

CODE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
echo $CODE_DIR

export PYTHONPATH=$PYTHONPATH:$CODE_DIR

pip install -r $CODE_DIR/requirements.txt --user

# download punkt, perluniprops
if [ ! -d "/usr/local/share/nltk_data/tokenizers/punkt" ]; then
    python2 -m nltk.downloader punkt
fi


# SQuAD preprocess is in charge of downloading
# and formatting the data to be consumed later
DATA_DIR=data
DOWNLOAD_DIR=download
mkdir -p $DATA_DIR
rm -rf $DATA_DIR
python2 $CODE_DIR/preprocessing/squad_preprocess.py

# Download distributed word representations
python2 $CODE_DIR/preprocessing/dwr.py

# Data processing for TensorFlow
python2 $CODE_DIR/qa_data.py --glove_dim 100
