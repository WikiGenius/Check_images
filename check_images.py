#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# */AIPND/intropylab-classifying-images/check_images.py
#
# TODO: 0. Fill in your information in the programming header below
# PROGRAMMER: Muhammed El-Yamani
# DATE CREATED: 11/6/2020
# REVISED DATE:             <=(Date Revised - if any)
# REVISED DATE: 18/6/2020 - added import statement that imports the print
#                           functions that can be used to check the lab
# PURPOSE: Check images & report results: read them in, predict their
#          content (classifier), compare prediction to actual value labels
#          and output results
#
# Use argparse Expected Call with <> indicating expected user input:
#      python check_images.py --dir <directory with images> --arch <model>
#             --dogfile <file that contains dognames>
#   Example call:
#    python check_images.py --dir pet_images/ --arch vgg --dogfile dognames.txt
##

# Imports python modules
from argparse import ArgumentParser
from time import time, sleep
from os import listdir
import datetime
# Imports classifier function for using CNN to classify images
from classifier import classifier

# Imports print functions that check the lab
from print_functions_for_lab_checks import *

# Main program function defined below


def main():
    # TODO: 1. Define start_time to measure total program runtime by
    # collecting start time
    start_time = time()

    # TODO: 2. Define get_input_args() function to create & retrieve command
    # line arguments
    in_arg = get_input_args()

    # TODO: 3. Define get_pet_labels() function to create pet image labels by
    # creating a dictionary with key=filename and value=file label to be used
    # to check the accuracy of the classifier function
    answers_dic = get_pet_labels(in_arg.dir)

    # TODO: 4. Define classify_images() function to create the classifier
    # labels with the classifier function using in_arg.arch, comparing the
    # labels, and creating a dictionary of results (result_dic)
    result_dic = classify_images(in_arg.dir, answers_dic, in_arg.arch)

    # TODO: 5. Define adjust_results4_isadog() function to adjust the results
    # dictionary(result_dic) to determine if classifier correctly classified
    # images as 'a dog' or 'not a dog'. This demonstrates if the model can
    # correctly classify dog images as dogs (regardless of breed)
    adjust_results4_isadog(result_dic, in_arg.dogfile)

    # TODO: 6. Define calculates_results_stats() function to calculate
    # results of run and puts statistics in a results statistics
    # dictionary (results_stats_dic)
    results_stats_dic = calculates_results_stats(result_dic)

    # TODO: 7. Define print_results() function to print summary results,
    # incorrect classifications of dogs and breeds if requested.
    print_results(result_dic, results_stats_dic, in_arg.arch,
                  print_incorrect_dogs=True, print_incorrect_breed=True)
    # TODO: 1. Define end_time to measure total program runtime
    # by collecting end time
    end_time = time()

    # TODO: 1. Define tot_time to computes overall runtime in
    # seconds & prints it in hh:mm:ss format
    tot_time = end_time - start_time
    print("\n** Total Elapsed Runtime:",
          str(datetime.timedelta(seconds=tot_time)))


# TODO: 2.-to-7. Define all the function below. Notice that the input
# parameters and return values have been left in the function's docstrings.
# This is to provide guidance for achieving a solution similar to the
# instructor provided solution. Feel free to ignore this guidance as long as
# you are able to achieve the desired outcomes with this lab.

def get_input_args():
    """
    Retrieves and parses the command line arguments created and defined using
    the argparse module. This function returns these arguments as an
    ArgumentParser object.
     3 command line arguments are created:
       dir - Path to the pet image files(default- 'pet_images/')
       arch - CNN model architecture to use for image classification(default-
              pick any of the following vgg, alexnet, resnet)
       dogfile - Text file that contains all labels associated to dogs(default-
                'dognames.txt'
    Parameters:
     None - simply using argparse module to create & store command line arguments
    Returns:
     parse_args() -data structure that stores the command line arguments object
    """
    # create ArgumentParser object named parser
    parser = ArgumentParser()

    # Arg1: Path to the pet image files
    parser.add_argument('--dir', type=str, default='./pet_images/',
                        help='Path to the pet image files')
    # Arg2: Type of CNN model architecture to use for image classification
    parser.add_argument('--arch', type=str, default='vgg',
                        help='Type of CNN model architecture to use for image classification(vgg, alexnet, resnet)')
    # Arg3: Path Text file that contains all labels associated to dogs
    parser.add_argument('--dogfile', type=str, default='./dognames.txt',
                        help='Path Text file that contains all labels associated to dogs')
    # return  parsed argument as collection
    return parser.parse_args()


def get_pet_labels(image_dir: str):
    """
    Creates a dictionary of pet labels based upon the filenames of the image
    files. Reads in pet filenames and extracts the pet image labels from the
    filenames and returns these labels as petlabel_dic. This is used to check
    the accuracy of the image classifier model.
    Parameters:
     image_dir - The (full) path to the folder of images that are to be
                 classified by pretrained CNN models (string)
    Returns:
     petlabels_dic - Dictionary storing image filename (as key) and Pet Image
                     Labels (as value)
    """
    # create dictionary as petlabel_dic to store filename (as key) and Pet Image
    # Labels (as value)
    petlabel_dic = dict()

    # the file names in the given directory
    in_files = listdir(image_dir)

    # process over each file
    for in_file in in_files:
        # skip file names that starts with .(like .Ds_store)
        if in_file[0] == '.':
            continue

        # Extract the pet image label from file name in format
        pet_label = in_file.replace('.jpg', '').replace(
            '_', ' ').strip('0123456789 ').lower()

        # skip duplicate files
        if in_file not in petlabel_dic:
            petlabel_dic[in_file] = pet_label
        else:
            print("Warning: Duplicate file names")

    # Return dictionary of {key = file name : value = pet_label}
    return petlabel_dic


def classify_images(images_dir: str, petlabel_dic: dict, model: str):
    """
    Creates classifier labels with classifier function, compares labels, and
    creates a dictionary containing both labels and comparison of them to be
    returned.
     PLEASE NOTE: This function uses the classifier() function defined in
     classifier.py within this function. The proper use of this function is
     in test_classifier.py Please refer to this program prior to using the
     classifier() function to classify images in this function.
     Parameters:
      images_dir - The (full) path to the folder of images that are to be
                   classified by pretrained CNN models (string)
      petlabel_dic - Dictionary that contains the pet image(true) labels
                     that classify what's in the image, where its key is the
                     pet image filename & its value is pet image label where
                     label is lowercase with space between each word in label
      model - pretrained CNN whose architecture is indicated by this parameter,
              values must be: resnet alexnet vgg (string)
     Returns:
      results_dic - Dictionary with key as image filename and value as a List
             (index)idx 0 = pet image label (string)
                    idx 1 = classifier label (string)
                    idx 2 = 1/0 (int)   where 1 = match between pet image and
                    classifer labels and 0 = no match between labels
    """
    # Dictionary with key as image filename and value as a List[idx 0,idx 1,idx 2]
    # format-> {key=image filename:value=[pet image label(string),
    # classifier label(string), 1/0 (int)]}
    results_dic = dict()

    # Iterate over all image files
    for img_name in petlabel_dic:

        # image path
        img_path = images_dir + img_name

        # Classify image using the chosen model (string)
        classifier_label = classifier(img_path, model)

        # Format classifier_label
        classifier_label = classifier_label.lower().strip()

        # Extract truth pet label
        truth = petlabel_dic[img_name]

        # Compare between truth and classifier_label
        # Find start idx truth in classifier_label string if existed
        found_idx = classifier_label.find(truth)

        # Additional Condition to improve matching

        # Condition to make sure the found index starts and not part in another word
        Cond_found_begin_word = (
            found_idx == 0 or classifier_label[found_idx-1] == ' ')

        # Condition to make sure the word ends and not part in another word
        Cond_found_end_word = (len(classifier_label) == found_idx + len(truth) or
                               classifier_label[found_idx+len(truth): found_idx+len(truth) + 1] in [' ', ','])

        # combine the conditions
        match = found_idx >= 0 and Cond_found_begin_word and Cond_found_end_word

        # Add the list into dictionary results_dic
        results_dic[img_name] = [truth, classifier_label, match]
    # Return
    return results_dic


def adjust_results4_isadog(results_dic: dict, dogsfile: str):
    """
    Adjusts the results dictionary to determine if classifier correctly
    classified images 'as a dog' or 'not a dog' especially when not a match.
    Demonstrates if model architecture correctly classifies dog images even if
    it gets dog breed wrong (not a match).
    Parameters:
      results_dic - Dictionary with key as image filename and value as a List
             (index)idx 0 = pet image label (string)
                    idx 1 = classifier label (string)
                    idx 2 = 1/0 (int)  where 1 = match between pet image and
                            classifer labels and 0 = no match between labels
                    --- where idx 3 & idx 4 are added by this function ---
                    idx 3 = 1/0 (int)  where 1 = pet image 'is-a' dog and
                            0 = pet Image 'is-NOT-a' dog.
                    idx 4 = 1/0 (int)  where 1 = Classifier classifies image
                            'as-a' dog and 0 = Classifier classifies image
                            'as-NOT-a' dog.
     dogsfile - A text file that contains names of all dogs from ImageNet
                1000 labels (used by classifier model) and dog names from
                the pet image files. This file has one dog name per line.
                Dog names are all in lowercase with spaces separating the
                distinct words of the dogname. This file should have been
                passed in as a command line argument. (string - indicates
                text file's name)
    Returns:
           None - results_dic is mutable data type so no return needed.
    """
    # Open file dogsfile
    with open(dogsfile, 'r') as file:
        # Exteract all lines into list
        dog_names = file.readlines()

    # iterate over all images in dictionary
    for img_name in results_dic:
        # Extract pet image label (truth)
        truth = results_dic[img_name][0]

        # initial condition
        truth_isadog = 0

        # iterate over all dog_names
        for dog_name in dog_names:
            # Make sure dog_name without any white space in leading and trail
            dog_name = dog_name.strip()

            # Compare truth with the dog_name in the dogsfile

            # Find start idx truth in classifier_label string if existed
            found_idx = dog_name.find(truth)

            # Additional Condition to improve matching

            # Condition to make sure the found index starts and not part in another word
            Cond_found_begin_word = (
                found_idx == 0 or dog_name[found_idx-1] == ' ')

            # Condition to make sure the word ends and not part in another word
            Cond_found_end_word = (len(dog_name) == found_idx + len(truth) or
                                   dog_name[found_idx+len(truth): found_idx+len(truth) + 1] in [' ', ','])

            # combine the conditions
            truth_isadog = found_idx >= 0 and Cond_found_begin_word and Cond_found_end_word

            # break the dognames loop if it is a dog
            if truth_isadog == 1:
                break
        # Append truth is a dog? idx 3 to results_dic
        results_dic[img_name] += [truth_isadog]

        ##########################################################

        # Extract classifier
        classifier = results_dic[img_name][1]

        # initial condition
        classifier_isadog = 0

        # iterate over all dog_names
        for dog_name in dog_names:
            # Make sure dog_name without any white space in leading and trail
            dog_name = dog_name.strip()

            # Compare classifier with the dog_name in the dogsfile

            # Find start idx classifier in classifier_label string if existed
            found_idx = dog_name.find(classifier)

            # Additional Condition to improve matching

            # Condition to make sure the found index starts and not part in another word
            Cond_found_begin_word = (
                found_idx == 0 or dog_name[found_idx-1] == ' ')

            # Condition to make sure the word ends and not part in another word
            Cond_found_end_word = (len(dog_name) == found_idx + len(classifier) or
                                   dog_name[found_idx+len(classifier): found_idx+len(classifier) + 1] in [' ', ','])

            # combine the conditions
            classifier_isadog = found_idx >= 0 and Cond_found_begin_word and Cond_found_end_word

            # break the dognames loop if it is a dog
            if classifier_isadog == 1:
                break

        # Append classifier is a dog? idx 4 to results_dic
        results_dic[img_name] += [classifier_isadog]


def calculates_results_stats(results_dic: dict):
    """
    Calculates statistics of the results of the run using classifier's model
    architecture on classifying images. Then puts the results statistics in a
    dictionary (results_stats) so that it's returned for printing as to help
    the user to determine the 'best' model for classifying images. Note that
    the statistics calculated as the results are either percentages or counts.
    Parameters:
      results_dic - Dictionary with key as image filename and value as a List
             (index)idx 0 = pet image label (string)
                    idx 1 = classifier label (string)
                    idx 2 = 1/0 (int)  where 1 = match between pet image and
                            classifer labels and 0 = no match between labels
                    idx 3 = 1/0 (int)  where 1 = pet image 'is-a' dog and
                            0 = pet Image 'is-NOT-a' dog.
                    idx 4 = 1/0 (int)  where 1 = Classifier classifies image
                            'as-a' dog and 0 = Classifier classifies image
                            'as-NOT-a' dog.
    Returns:
     results_stats - Dictionary that contains the results statistics (either a
                     percentage or a count) where the key is the statistic's
                     name (starting with 'pct' for percentage or 'n' for count)
                     and the value is the statistic's value
    """

    # Dictionary that contains the results statistics
    results_stats = dict()

    # Counts:
    # Number of Images
    n_images = len(results_dic)
    results_stats['n_images'] = n_images

    # Number of Dog Images
    n_dogs_img = sum([v[3] for v in results_dic.values()])
    results_stats['n_dogs_img'] = n_dogs_img

    # Number of "Not-a" Dog Images
    n_notdogs_img = n_images - n_dogs_img
    results_stats['n_notdogs_img'] = n_notdogs_img

    # Percentages:
    # % Correctly Classified Dog Images
    if n_dogs_img > 0:
        pct_correct_dogs = sum([
            sum(v[3:]) == 2 for v in results_dic.values()])/n_dogs_img * 100.0
        results_stats['pct_correct_dogs'] = pct_correct_dogs
    else:
        pct_correct_dogs = 0
    results_stats['pct_correct_dogs'] = pct_correct_dogs

    # % Correctly Classified "Not-a" Dog Images
    if n_notdogs_img > 0:
        pct_correct_notdogs = sum([
            sum(v[3:]) == 0 for v in results_dic.values()])/n_notdogs_img * 100.0

    else:
        pct_correct_notdogs = 0
    results_stats['pct_correct_notdogs'] = pct_correct_notdogs

    # % Correctly Classified Breeds of Dog Images
    if n_dogs_img > 0:
        pct_correct_breed = sum([
            sum(v[2:]) == 3 for v in results_dic.values()])/n_dogs_img * 100.0
        results_stats['pct_correct_breed'] = pct_correct_breed
    else:
        pct_correct_breed = 0
    results_stats['pct_correct_breed'] = pct_correct_breed

    # % Label matches Images (option)
    pct_label_match = sum([
        v[2] for v in results_dic.values()])/n_images * 100.0
    results_stats['pct_label_match'] = pct_label_match
    return results_stats


def print_results(results_dic: dict, results_stats: dict, model: str, print_incorrect_dogs=False, print_incorrect_breed=False):
    """
    Prints summary results on the classification and then prints incorrectly
    classified dogs and incorrectly classified dog breeds if user indicates
    they want those printouts (use non-default values)
    Parameters:
      results_dic - Dictionary with key as image filename and value as a List
             (index)idx 0 = pet image label (string)
                    idx 1 = classifier label (string)
                    idx 2 = 1/0 (int)  where 1 = match between pet image and
                            classifer labels and 0 = no match between labels
                    idx 3 = 1/0 (int)  where 1 = pet image 'is-a' dog and
                            0 = pet Image 'is-NOT-a' dog.
                    idx 4 = 1/0 (int)  where 1 = Classifier classifies image
                            'as-a' dog and 0 = Classifier classifies image
                            'as-NOT-a' dog.
      results_stats - Dictionary that contains the results statistics (either a
                     percentage or a count) where the key is the statistic's
                     name (starting with 'pct' for percentage or 'n' for count)
                     and the value is the statistic's value
      model - pretrained CNN whose architecture is indicated by this parameter,
              values must be: resnet alexnet vgg (string)
      print_incorrect_dogs - True prints incorrectly classified dog images and
                             False doesn't print anything(default) (bool)
      print_incorrect_breed - True prints incorrectly classified dog breeds and
                              False doesn't print anything(default) (bool)
    Returns:
           None - simply printing results.
    """

    # print model
    print("\n\n***Results summary for Pretrained CNN model: %s***" % model.upper())

    # Extract statistics
    n_images = results_stats['n_images']
    n_dogs_img = results_stats['n_dogs_img']
    n_notdogs_img = results_stats['n_notdogs_img']

    pct_correct_dogs = results_stats['pct_correct_dogs']
    pct_correct_notdogs = results_stats['pct_correct_notdogs']
    pct_correct_breed = results_stats['pct_correct_breed']
    pct_label_match = results_stats['pct_label_match']

    # print Statistics
    print()
    print('Statistics summary:')
    print("N Images: %2d\nN Dog Images: %2d\nN NotDog Images: %2d\n\nPct Corr dog: %5.1f\nPct Corr NOTdog: %5.1f\nPct Corr Breed: %5.1f\nPct Corr Label Match: %5.1f\n\n"
          % (n_images, n_dogs_img, n_notdogs_img, pct_correct_dogs, pct_correct_notdogs, pct_correct_breed, pct_label_match))

    # print Misclassified Dogs
    if print_incorrect_dogs == True:
        n_correct_dogs = int(pct_correct_dogs * n_dogs_img / 100.0)
        n_correct_notdogs = int(pct_correct_notdogs * n_notdogs_img / 100.0)

        if n_correct_dogs + n_correct_notdogs != n_images:
            print('################################################################')
            print()

            n_Misclassified_Dogs = 0
            for key in results_dic:
                if sum(results_dic[key][3:]) == 1:

                    # print Misclassified Dogs files
                    if n_Misclassified_Dogs == 0:
                        print('Misclassified Dogs files: ')
                    print('Pet image Label: %20s   Classifier: %20s' %
                          (results_dic[key][0], results_dic[key][1]))

                    # count n_Misclassified Dogs
                    n_Misclassified_Dogs += 1

            # print n_Misclassified Dogs
            print("n_Misclassified_Dogs: %d" % n_Misclassified_Dogs)

    # print Misclassified Breed's of Dog
    if print_incorrect_breed == True:
        n_correct_dogs = int(pct_correct_dogs * n_dogs_img / 100.0)
        n_correct_breed = int(pct_correct_breed * n_dogs_img / 100.0)
        if n_correct_dogs != n_correct_breed:
            print('################################################################')
            print()
            n_Misclassified_Breed_Dogs = 0
            for key in results_dic:
                if sum(results_dic[key][3:]) == 2 and results_dic[key][2] == 0:

                    # print Misclassified Breed's of Dog files
                    if n_Misclassified_Breed_Dogs == 0:
                        print("Misclassified Breed's of Dog files")
                    print('Pet image Label: %20s   Classifier: %20s' %
                          (results_dic[key][0], results_dic[key][1]))

                    # count n_Misclassified Breed's of Dog
                    n_Misclassified_Breed_Dogs += 1

            # print n_Misclassified Breed's of Dog
            print("n_Misclassified_Breed_Dogs: %d" %
                  n_Misclassified_Breed_Dogs)


# Call to main function to run the program
if __name__ == "__main__":
    main()
