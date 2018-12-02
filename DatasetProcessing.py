import os
from collections import Counter
from sklearn.utils import Bunch
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk import WordNetLemmatizer
import numpy as np

"""
This file will create a dictionary of a given dataset and then
prepare the dataset for use with an sklearn classifier
"""


def create_dictionary(directory: str):
    """
    This function creates a dictionary using the files provided in the dataset. This uses a Counter object
    that was much harder to work with than I had anticipated thus it is converted later to simplify the process.

    If I was to do this over again I wouldn't use this "Counter" again, it would be much simpler to just create a
    dictionary and use the word as the key and the count as the value. From there you could just loop through every
    email and increment the value whenever you encounter the same word or add a new one initialized to 1.
    It works but I'm not happy with it

    :param directory: The directory to read from
    :return converted: The final converted dictionary
    """
    found_words = []
    list_emails = [os.path.join(directory, f) for f in os.listdir(directory)]
    for i in list_emails:
        with open(i) as m:
            for j, line in enumerate(m):
                words = line.split()
                if words != "Subject:" or "Subject":
                    found_words += words
    dictionary_dirty = Counter(found_words)  # Creating a counter object
    dictionary_clean = clean_punctuation(dictionary_dirty)
    dictionary_clean = dictionary_clean.most_common(5000)  # counting the top 5000 words found.
    converted = counter_converter(dictionary_clean)
    return converted


def clean_punctuation(data: dict):
    """
    This function takes the created dictionary and removes all of the non alphabetic
    symbols and characters from the dictionary not the best way of doing it but it works for the data I'm dealing with.

    :param data: The dictionary to be cleaned
    :return data: The cleaned dictionary
    """
    exclusion_list = data.keys()
    for item in list(exclusion_list):
        if not item.isalpha():  # checking so see if the data is alphabetic
            del data[item]
        if len(item) == 1:  # checking to see if the "word" is a single character
            del data[item]
    return data


def counter_converter(dictionary: list):
    """
    This converts the dictionary created using the Counter object into a usable
    dictionary for use later in the process. Not the best way to handle the dictionary but
    for this it will do

    :param dictionary: The extracted dictionary from the Counter
    :return usable_dict: The converted usable dictionary
    """
    usable_dict = {}
    for (word, count) in dictionary:
        usable_dict[word] = count
    return usable_dict


def process_files(directory: str, dictionary: dict):
    """
    This function processes all of the files in a given directory, it scans for any and all files
    in the directory and feeds them to the process_data function that extracts the data from the
    individual emails.

    :param directory: The directory of the stored files
    :param dictionary: The dictionary used to count word frequency
    :return: A Bunch containing all of the extracted data to be used with the classifiers
    """
    capture_data = []
    determined_targets = []  # what this email is by default is is flagged as non spam
    target_labels = ["non-spam", "spam"]
    feature_names = dictionary.keys()  # change this maybe
    list_emails = [os.path.join(directory, f) for f in os.listdir(directory)]
    for emails in list_emails:
        print('\r' + str(round(list_emails.index(emails)/(len(list_emails) - 1) * 100, 1)) + "% processed", end='')
        extracted_data, extracted_target = process_data(emails, dictionary)
        capture_data.append(extracted_data)
        determined_targets.append(extracted_target)
    return Bunch(data=capture_data, target=determined_targets, target_labels=target_labels, feature_names=feature_names)


def process_data(current_email: str, dictionary: dict):
    """
    This function takes the individual emails extracted from the last function and extracts the data from the
    file. It counts the frequency of each word in a specific email with respect to the dictionary.

    :param current_email: The current email being read
    :param dictionary: The dictionary to compare
    :return text_data / determined_target: The frequency of each word / The type of email (spam/non-spam)
    """
    text_data = np.zeros((len(dictionary)), dtype=np.int)
    determined_target = 0
    if "spmsg" in current_email:
        determined_target = 1
    current_text = open(current_email).read()
    data_collected = process_words(current_text)
    for current_word in data_collected:
        if current_word in list(dictionary.keys()):
            index = list(dictionary.keys()).index(current_word)
            text_data[index] += 1
        else:
            continue
    return text_data, determined_target


def process_words(data: str):
    """
    This method was developed after the main implementation took place and I don't have the time now to
    go and adapt all the code to use this properly. But this way is a really nice and simple way to tokenize, remove
    stopwords and lemmatize the data. I kinda wish I had found out how to do this earlier, but I know now.

    :param data: The data to be processed
    :return processed_data: cleaned data ready for use
    """
    processed_data = []

    stop_words = set(stopwords.words('english'))
    word_splitter = RegexpTokenizer(r'\w+')
    lemmatizer = WordNetLemmatizer()
    processing = word_splitter.tokenize(data)
    for i in processing:
        if i not in stop_words and i.isalpha() and i != "Subject":
            processed_data.append(i)
    for j in processed_data:
        lemmatizer.lemmatize(j)

    return processed_data


def process_dataset(directory: str):
    """
    This is the harness that will create the dictionary and process the emails it will also download any
    nltk packages that as missing from the host machine.

    :param directory: The directory that the data is to be read from
    :return processed_bunch: The final processed data stored in a Bunch for use with the classifiers
    """
    print("Checking for nltk package updates...")
    nltk.download('stopwords')
    nltk.download('wordnet')
    print("\nProcessing dataset from: " + directory)
    dictionary = create_dictionary(directory)
    processed_bunch = process_files(directory, dictionary)
    return processed_bunch
