import re


class Regex(object):
    """
    This class would take regex pattern list in order to apply multiple regular expressions on text
    :param regex_pattern_list: list of tuple in form [(regular_expression,text_to_replace),(.......),]
    :param compiled_patterns: list of all compiled objects of regular expressions
    :param processed_text: text that stores output

    for example:
        regex = Regex([(r'[^\w\s]',''),(r'[\d]','__number__')])
        regex.text_substitute("Hey, Are you coming today at 9?")
        output: "Hey Are you coming today at __number__"
    """

    def __init__(self, regex_pattern_list):
        self.regex_pattern_list = regex_pattern_list
        self.compiled_patterns = []
        self.text = None
        self.processed_text = None
        for pattern in self.regex_pattern_list:
            self.compiled_patterns.append(re.compile(pattern[0]))

    def text_substitute(self, text):
        self.text = text
        self.processed_text = text
        for pattern_index, compiled_pattern in enumerate(self.compiled_patterns):
            self.processed_text = compiled_pattern.sub(self.regex_pattern_list[pattern_index][1],
                                                       self.processed_text)
        return self.processed_text
