from pycorenlp import StanfordCoreNLP
from nltk.corpus import stopwords


class StanfordParser(object):
    """Stanford Parser for information extraction and phrase detection
    Start Stanford Server:
        cd <Stanford CoreNLP folder>
        java -Xmx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 15000
    """
    def __init__(self, properties=None):
        self.__properties = properties if properties is not None else \
            {'annotators': 'tokenize,ssplit,pos,lemma,ner,depparse,mention,dcoref,natlog,openie',
             'depparse.model': 'edu/stanford/nlp/models/parser/nndep/english_SD.gz',
             'outputFormat': 'json'}
        self.__nlp = StanfordCoreNLP('http://localhost:9000')
        self.__instance = None

        self.__stopwords = stopwords.words('english')
        with open('./data/pronouns.txt') as f:
            self.__pronouns = [word.strip() for word in f.readlines()]

    def annotate(self, text):
        self.__instance = self.__nlp.annotate(text, properties=self.__properties)

    def sentence_tokenize(self):
        """split text into sentences
        :return: list of sentences"""
        return [' '.join(words) for words in self.word_tokenize()]

    def word_tokenize(self):
        """split text into list of list of words
        :return: list of list of words"""
        words_list = []
        for sentence in self.__instance['sentences']:
            words_list.append([token['word'] for token in sentence['tokens']])
        return words_list

    def sentence_lemmatize(self):
        """:return: lemmatized sentences for a given text"""
        return [' '.join(lemmas) for lemmas in self.word_lemmatize()]

    def word_lemmatize(self):
        """:return: list of list of lemmatized words"""
        lemmas_list = []
        for sentence in self.__instance['sentences']:
            lemmas_list.append([token['lemma'] for token in sentence['tokens']])
        return lemmas_list

    def sentence_pos_tagging(self):
        """:return: list of sentences with pos tag for each word"""
        return [' '.join(tags) for tags in self.word_pos_tagging()]

    def word_pos_tagging(self):
        """:return: list of list of words with pos tag"""
        tags_list = []
        for sentence in self.__instance['sentences']:
            tags_list.append([token['word'] + '/' + token['pos'] for token in sentence['tokens']])
        return tags_list

    def dependency_triples(self):
        """here the enhanced++ dependencies model is used
        :return: list of list of dependency triples"""
        triples_list = []
        for sentence in self.__instance['sentences']:
            triples = [(dep['governorGloss'], dep['dependentGloss'], dep['dep'])
                       for dep in sentence['enhancedPlusPlusDependencies']]
            triples_list.append(triples)
        return triples_list

    def parsed_dependency(self):
        return [sent['parse'] for sent in self.__instance['sentences']]

    def coreference_chains(self):
        """get the coreference information from instance
        :return: coreference chains"""
        return [coref for _, coref in self.__instance['corefs'].items()]

    def simplified_coreference_chains(self):
        """get simplified coreference information from instance (text, sentNum, headIndex, startIndex, endIndex)
        :return: coreference chains"""
        simplified = []
        for chain in self.coreference_chains():
            simp = []
            for coref in chain:
                simp.append({'text': coref['text'], 'sentNum': coref['sentNum'], 'headIndex': coref['headIndex'],
                             'startIndex': coref['startIndex'], 'endIndex': coref['endIndex']})
            simplified.append(simp)
        return simplified

    def get_instance(self):
        return self.__instance

    def get_nlp(self):
        return self.__nlp

    """=====================================Below are the experimental functions====================================="""
    """=============================================================================================================="""
    def replace_coreference(self):
        """replace coref to original head words"""
        coref_chains = self.coreference_chains()
        words_list = self.word_tokenize()
        for chain in coref_chains:
            if self.__check_pronominal(chain):  # if a chain contains pronouns only or non-pronouns only, ignore it
                continue
            replace_token = ''
            for value in chain:
                if value['type'] != 'PRONOMINAL':
                    tokens = words_list[value['sentNum'] - 1][value['startIndex'] - 1: value['endIndex'] - 1]
                    tokens = [word for word in tokens
                              if word not in (self.__pronouns + self.__stopwords)]
                    replace_token = ' '.join(tokens)
                    break
            for value in chain:
                if value['type'] == 'PRONOMINAL':
                    words_list[value['sentNum'] - 1][value['headIndex'] - 1] = replace_token
        return [' '.join(words) for words in words_list]

    @staticmethod
    def __check_pronominal(coref_chain):
        pronoun = 0
        other = 0
        for val in coref_chain:
            if val['type'] == 'PRONOMINAL':
                pronoun += 1
            else:
                other += 1
        if pronoun == len(coref_chain) or other == len(coref_chain):
            return True
        else:
            return False
