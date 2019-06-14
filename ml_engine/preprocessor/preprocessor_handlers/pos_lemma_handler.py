from .base_handler import BaseTextProcessing
from .utils import remove_common_item_from_lists
from ml_engine.config import ml_logger
from .constants import lemmatizer, part_of_speech


class POSLemmaHandler(BaseTextProcessing):
    """
    This is a POS Lemma handler, which uses to find the different part of speech in text.
    This class would first find the specific POS taggers and then apply the lemmatization on
    the resulting pos taggers that depends on whatever flag sets in request.

    if flag_pos_lemma is set,

      and if flag_tokenize is not set,
         then first convert text_to_process into tokens and then find pos tagging

      and if flag_tokenize is set,
         then find pos tagging.

      once pos tagging is found,
        then convert those taggers into its lemma tokens

    if flag_stem_without_stop_word is set,
       remove all the stop words from pos_lemma_tokens

    Output:
        tokens = ['ate','eating']
      After pos_lemma i.e flag_pos_lemma is set,
        kwargs['pos_lemma'] = ['ate','eat']

    """

    def convert_tokens_to_lemma(self, tokens):
        return lemmatizer.lemma_tokens(tokens)

    def pos_lemma_handling(self, text_to_process, tokens=None):

        pos_lemma_tokens = []
        pos_lemma_tokens_without_stopwords = []
        pos_lemma_dict = dict()

        if self.flag_pos_lemma:
            if self.flag_tokenize:
                pos_tokens = self.pos_tagger(tokens)
            else:
                pos_tokens = self.pos_tagger(self.tokenizer.tokenize(text_to_process))

            pos_lemma_tokens = self.convert_tokens_to_lemma(
                [pos_token[0] for pos_token in pos_tokens if pos_token[1] in part_of_speech]
            )
        if self.flag_pos_lemma_without_stopwords and pos_lemma_tokens:
            pos_lemma_tokens_without_stopwords = remove_common_item_from_lists(pos_lemma_tokens, self.stop_words_list)
        elif self.flag_pos_lemma_without_stopwords and not pos_lemma_tokens:
            pos_lemma_tokens_without_stopwords = remove_common_item_from_lists(
                self.convert_tokens_to_lemma(self.tokenizer.tokenizer(text_to_process)), self.stop_words_list
            )
        pos_lemma_dict['pos_lemma_tokens'] = pos_lemma_tokens
        pos_lemma_dict['pos_lemma_tokens_without_stopwords'] = pos_lemma_tokens_without_stopwords

        return pos_lemma_dict

    def _handle_processing(self, **kwargs):
        pos_lemma_dict = self.pos_lemma_handling(kwargs['text_to_process'], kwargs['tokens'])
        kwargs.update(pos_lemma_dict)

        ml_logger.debug("POS Lemma handler ==>> pos_lemma_tokens:{} and pos_lemma_tokens_without_stopwords:{}".format(
            kwargs['pos_lemma_tokens'], kwargs['pos_lemma_tokens_without_stopwords']
        ))
        return kwargs
