from .base_handler import BaseTextProcessing
from .constants import regex_to_remove_punctuations
from ml_engine.nlp.regex import Regex
from ml_engine.config import ml_logger


class PunctuationHandler(BaseTextProcessing):
    """
    This class would act as a preprocessor handlers which removes all the punctuations
    from the text using regex,and update the output in kwargs dict.

    punctuations such as '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~' are removed if
    flag_punctuation_removal is set else no processing would occur on text
    input:
        text = "Hey, Can i come in?"
        text_to_process = "hey, can i come in?"
    output:
        if flag_punctuation_removal is set, "hey can i come in"
        else: "hey, can i come in?"
    """

    def remove_punctuation_from_text(self, text_to_process):
        regex_to_process = Regex(regex_to_remove_punctuations)
        text_with_punctuations_removed = regex_to_process.text_substitute(text_to_process)
        return text_with_punctuations_removed

    def _handle_processing(self, **kwargs):
        text_to_process = self.text.lower()
        if self.flag_punctuation_removal:
            text_with_punctuations_removed = self.remove_punctuation_from_text(text_to_process)
            kwargs.update({'text_to_process': text_with_punctuations_removed})
        else:
            kwargs.update({'text_to_process': text_to_process})
        ml_logger.debug("Punctuation handler ==>> text to process:{}".format(kwargs['text_to_process']))
        return kwargs
