#!/bin/bash
set -e

echo "==========================================================="
echo "Downloading Real-Pi Dataset (soldering_2)"
echo "==========================================================="

DRIVE_FILE_ID="https://drive.google.com/drive/folders/1eepafGELvc_7SHXhLbE0VHEzahou0CCk?usp=sharing"
OUTPUT_FILE="data.zip"

echo "Downloading from Google Drive..."
gdown --id $DRIVE_FILE_ID -O $OUTPUT_FILE

echo "Unzipping dataset..."
unzip -q $OUTPUT_FILE -d data/
rm $OUTPUT_FILE

echo "==========================================================="
echo "Dataset successfully downloaded and extracted into data/"
echo "==========================================================="
