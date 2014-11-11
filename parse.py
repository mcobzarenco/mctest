#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function, division
import argparse
import json
import errno
import os
import sys
from base64 import urlsafe_b64decode as b64decode, \
    urlsafe_b64encode as b64encode
from functools import partial
from itertools import imap, ifilter

import nltk
import pandas as pd
from google.protobuf import text_format

from mctest_pb2 import AnswerAsWords, QuestionAsWords, StoryAsWords, \
    AnswerAsEmbeddings, QuestionAsEmbeddings, StoryAsEmbeddings


DEFAULT_OUTPUT_FORMAT = 'json'

COLUMNS = ['id', 'description', 'story',
           'q1', 'a11', 'a12', 'a13', 'a14',
           'q2', 'a21', 'a22', 'a23', 'a24',
           'q3', 'a31', 'a32', 'a33', 'a34',
           'q4', 'a41', 'a42', 'a43', 'a44']
QUESTION_TYPES = ['one', 'multiple']

PUNCTS = ['.', '?', ',', '!', '"', '\'', '$', '%', '^', '&']


def question_text(question):
    return question.split(':')[1].strip()


def question_type(question):
    question_type, _ = question.split(':')
    assert question_type in QUESTION_TYPES
    return question_type


def row_to_dict(row, tokenize=None):
    return {
        'id': row['id'],
        'description': row['description'],
        'passage': tokenize(row['story']),
        'questions': [{
            'tokens': tokenize(question_text(row['q%d' % q_number])),
            'answers': [tokenize(row['a%d%d' % (q_number, a_number)])
                        for a_number in xrange(1, 5)],
            'type': question_type(row['q%d' % q_number])
        } for q_number in xrange(1, 5)]
    }



def datapoint_to_proto_as_words(datapoint):
    story = StoryAsWords()
    story.id = datapoint['id']
    story.description = datapoint['description']
    story.passage.extend(datapoint['passage'])
    for question_dict in datapoint['questions']:
        question = story.questions.add()
        if question_dict['type'] == 'one':
            question.type = QuestionAsWords.ONE
        elif question_dict['type'] == 'multiple':
            question.type = QuestionAsWords.MULTIPLE
        else:
            print('Invalid question type: %s' % question_dict['type'],
                  file=sys.stderr)
            sys.exit(3)
        question.tokens.extend(question_dict['tokens'])
        for answer_list in question_dict['answers']:
            answer = question.answers.add()
            answer.tokens.extend(answer_list)
    return story


def datapoint_to_proto_as_embeddings(datapoint):
    story = StoryAsEmbeddings()
    story.id = datapoint['id']
    story.description = datapoint['description']
    for passage_vec in datapoint['passage']:
        embed = story.passage.add()
        embed.value.extend(list(passage_vec))
    for question_dict in datapoint['questions']:
        question = story.questions.add()
        if question_dict['type'] == 'one':
            question.type = QuestionAsWords.ONE
        elif question_dict['type'] == 'multiple':
            question.type = QuestionAsWords.MULTIPLE
        else:
            print('Invalid question type: %s' % question_dict['type'],
                  file=sys.stderr)
            sys.exit(3)
        for token_vec in question_dict['tokens']:
            question.tokens.add().value.extend(list(token_vec))
        for answer_list in question_dict['answers']:
            answer = question.answers.add()
            for answer_vec in answer_list:
                answer.tokens.add().value.extend(list(answer_vec))
    return story


def tokens_to_embeddings(model, tokens):
    embeds = []
    for token in tokens:
        try:
            token = token.lower()
            embeds.append(model[token])
        except KeyError as e:
            print('WARNING: "%s" missing from vocabulary.' % token,
                  file=sys.stderr)
    return embeds


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Converts raw TSV files from the MCTest dataset')
    _arg = parser.add_argument
    _arg('-o', type=str, action='store', metavar='FORMAT',
         default=DEFAULT_OUTPUT_FORMAT,
         help='Output format: json, proto, prototext (default=%s)' %
         DEFAULT_OUTPUT_FORMAT)
    _arg('--rm-stop', type=str, action='store', metavar='FILE',
         help='Remove stop words specified by file (one word per line).')
    _arg('--rm-punct', action='store_true',
         help='Remove punctuation when tokenizing.')
    _arg('--model-file', type=str, action='store', metavar='FILE', default=None,
         help='File with word2vec model. If provided, makes it output' \
         'embeddings.')
    _arg('-i', type=str, action='store', metavar='FILE', default=None,
         help='Input file (TSV).')
    args = parser.parse_args()

    token_mappers = []

    if args.rm_stop:
        stopwords = open(args.rm_stop, 'r').read().split('\n')
        stopwords = set(map(lambda x: x.strip().rstrip(), stopwords))
        token_mappers.append(lambda x: x if x.lower() not in stopwords else None)

    if args.rm_punct:
        token_mappers.append(lambda x: x if x not in PUNCTS else None)

    as_embeddings = args.model_file is not None
    if args.model_file:
        import word2vec
        embedding_model = word2vec.load(args.model_file)
        def to_embeddings(token):
            try:
                return embedding_model[token.lower()]
            except KeyError as e:
                print('WARNING: "%s" missing from vocabulary.' % token,
                      file=sys.stderr)
            return None
        token_mappers.append(to_embeddings)

    def tokenize(text):
        if not isinstance(text, basestring):
            text = str(text)
        text = text.replace('\\newline', ' ')
        mapped = nltk.word_tokenize(text)
        for mapper in token_mappers:
            mapped = filter(lambda x: x is not None, map(mapper, mapped))
        return mapped

    data_in = open(args.i, 'r') if args.i else sys.stdin
    df = pd.read_csv(data_in, sep='\t', names=COLUMNS)
    for row in (df.ix[i] for i in df.index):
        datapoint = row_to_dict(row, tokenize)
        try:
            serialized = None
            if args.o == 'json':
                serialized = json.dumps(datapoint)
            elif args.o == 'proto' or args.o == 'prototext':
                proto = datapoint_to_proto_as_embeddings(datapoint) \
                        if as_embeddings else \
                           datapoint_to_proto_as_words(datapoint)
                serialized = proto.SerializeToString() if args.o == 'proto' \
                             else text_format.MessageToString(proto)
            else:
                print('Unknown output format "%s"' % args.o,
                      file=sys.stderr)
                sys.exit(2)
            assert serialized
            print(serialized, file=sys.stdout)
        except IOError as e:
            if e.errno == errno.EPIPE:
                sys.exit(0)
            raise e
