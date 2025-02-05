#!/bin/bash
mkdir -p dataset/
cd dataset/

echo "The datasets will be stored in the 'dataset' folder\n"

echo "Downloading KIT dataset"
gdown "https://drive.google.com/uc?id=11rFFjSyuvveElHyvUIvyV6_lLy4mUN-b"
echo "Extracting KIT dataset"
unzip KIT.zip -d KIT-ML
echo "Cleaning\n"
rm KIT.zip

echo "Downloading done!"
