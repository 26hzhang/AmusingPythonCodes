# encoding: utf-8
from stanford_corenlp_demo.stanford_corenlp import StanfordParser
from stanford_corenlp_demo.common import read_nth_data
from pprint import pprint

text, _ = read_nth_data('./data/train-data.xml', 6)
print(text + '\n')

parser = StanfordParser()  # create stanford corenlp parser
parser.annotate(text)  # annotate given text
print(parser.get_instance())  # print annotate text
print()

sentences = parser.sentence_tokenize()  # get sentences
rep_coref_sentences = parser.replace_coreference()
for sentence, rep_sentence in zip(sentences, rep_coref_sentences):
    print(sentence)
    print(rep_sentence + '\n')
print()

words_list = parser.word_tokenize()  # get tokenized words
for words in words_list:
    print(words)
print()

coref_chains = parser.coreference_chains()
for coref_chain in coref_chains:
    pprint(coref_chain)
    print()
