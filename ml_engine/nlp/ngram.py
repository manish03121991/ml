class Ngram(object):
    """
    This class would convert word list into ngram and output it in the grams of list.
    Stop words list is used to remove the stop words from ngram data.

    Attributes:
        gram: it is the number of grams which we want to convert into.
        word_list: list of words to be processed

    output:
        ngram = Ngrams()
        ngram.ngram_data(2,['my','name','is','bharat'])
        ['my name','name is','is bharat']

        if stop word list: ['is','am','a','are']

        ngram.ngram_data(2,['you','are','a','good','person'])
        ['you are','a good','good person']

    """

    @staticmethod
    def ngram_data(gram, word_list, stop_word_list):
        gram_list = list()
        for word_index in range(0, len(word_list) - gram + 1):
            gram_count = 0
            gram_string = ''
            gram_index = word_index
            stop_word_count = 0
            while gram_count < gram:
                if stop_word_list and (word_list[gram_index] in stop_word_list):
                    stop_word_count += 1
                gram_string += ' ' + word_list[gram_index]
                gram_count += 1
                gram_index += 1
            if stop_word_count < gram:
                gram_list.append(gram_string.strip())
        return gram_list
