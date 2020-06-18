#!/bin/sh
# */AIPND/intropylab-classifying-images/run_models_batch.sh
#                                                                             
# PROGRAMMER: Muhammed El-Yamani
# DATE CREATED: 11/6/2020                                  
# REVISED DATE: 18/6/2020  - reduce scope of program
# PURPOSE: Runs all three models to test which provides 'best' solution.
#          Please note output from each run has been piped into a text file.
#
# Usage: sh run_models_batch.sh    -- will run program from commandline
#  
python check_images.py --dir pet_images/ --arch resnet  --dogfile dognames.txt > resnet.txt
python check_images.py --dir pet_images/ --arch alexnet --dogfile dognames.txt > alexnet.txt
python check_images.py --dir pet_images/ --arch vgg  --dogfile dognames.txt > vgg.txt
