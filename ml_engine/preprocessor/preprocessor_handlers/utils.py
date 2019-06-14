import csv
from ml_engine.config import ml_logger

def read_data_from_file(filename, stemmer=None, lowercase=True):
    """
    This function would read the csv file and store the data in list.

    In our case, we are using this utility to read stop words from file
    and convert it into python list.

    :param filename: Name of the file with its absolute path
    :param stemmer: object of stemmer class
    :param lowercase: bool if to convert token from file to lowercase
    :return: list of tokens read from file
    """
    stop_words_list = []
    try:
        with open(filename, 'r+') as csvfile:
            file_reader = csv.reader(csvfile, delimiter='\n')
            for row in file_reader:
                token = row[0].lower() if lowercase else row[0]
                if stemmer:
                    stop_words_list.append(stemmer.stem_word(token))
                else:
                    stop_words_list.append(token)
    except Exception as e:
        ml_logger.debug("File not found in case of stop words:---{}".format(filename))
    return stop_words_list


def remove_common_item_from_lists(list1, list2):
    """
    This utility function would remove the common items of two lists

    such as
      list1: ['hello','hey','bye','take care']
      list2: ['hey','bye']
      output: ['hello','take care']

    :param list1: first list of some items
    :param list2: second list of some items
    :return: unique items from list1 and list2
    """
    return [elem for elem in list1 if elem not in list2]
