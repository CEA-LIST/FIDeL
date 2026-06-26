#!/bin/bash
set -e

echo "==========================================================="
echo "Downloading Real-Pi Dataset (soldering_2)"
echo "==========================================================="

DRIVE_FILE_ID="https://drive.google.com/file/d/1XPgYZVfaYmQ1UjWWLLsdXj6I09h219gq/view?usp=sharing"
OUTPUT_FILE="data.zip"

echo "Downloading from Google Drive..."
gdown $DRIVE_FILE_ID -O $OUTPUT_FILE

echo "Unzipping dataset..."
unzip -q $OUTPUT_FILE -d .
rm $OUTPUT_FILE

echo "==========================================================="
echo "Dataset successfully downloaded and extracted into data/"
echo "==========================================================="
