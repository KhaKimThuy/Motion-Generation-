#!/bin/bash
mkdir -p dataset/
cd dataset/

echo "The datasets will be stored in the 'dataset' folder\n"

echo "Downloading HUMANML3D dataset"
gdown "https://drive.google.com/uc?id=1O0NC4-5qzHu6yUL5i7bs_hRP6-aHNgCZ"
echo "Extracting HUMANML3D dataset"
unzip HUMANML3D.zip -d HumanML3D
echo "Cleaning\n"
rm HUMANML3D.zip

echo "Downloading done!"
