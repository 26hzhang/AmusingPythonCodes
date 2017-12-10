# encoding: utf-8
import xml.etree.ElementTree as ETree
import requests
import re


def dataset_xml_iterator(filename):
    """An iterator to convert xml-format dataset to more readable text format"""
    instances = ETree.parse(filename).getroot()
    for instance in instances:
        paragraph = instance.find('text').text
        questions = instance.findall('questions')[0]
        queries = []
        for question in questions.findall('question'):
            tmp_dict = {'Text': question.get('text')}
            for answer in question.findall('answer'):
                tmp_dict[answer.get('correct')] = answer.get('text')
            queries.append(tmp_dict)
        yield paragraph, queries


def read_nth_data(filename, n):
    """Read Nth paragraph and corresponding queries"""
    index = 0
    for paragraph, queries in dataset_xml_iterator(filename):
        index += 1
        if n == index:
            # para = paragraph
            # que = queries
            return paragraph, queries
    return None


def extract_conceptnet(phrase):
    """Access ConceptNet API and read relational triples as well as their weight and simple example"""
    url_head = 'http://api.conceptnet.io/c/en/'  # access ConceptNet API
    raw_json = requests.get(url_head + phrase).json()
    edges = raw_json['edges']
    if not edges:  # if edges is empty, which means ConceptNet doesn't contain such concept or node
        return None
    concepts = []
    for edge in edges:
        triple = re.findall(r'/a/\[/r/(.*?)/.*?,/c/en/(.*?)/.*?,/c/en/(.*?)/.*?\]', edge['@id'])[0]  # ERE triple
        surface_text = re.sub(r'[\[\]]', '', '' if edge['surfaceText'] is None else edge['surfaceText'])  # example
        weight = edge['weight']  # weight
        concepts.append({'Triple': triple, 'weight': weight, 'example': surface_text})
    return concepts



