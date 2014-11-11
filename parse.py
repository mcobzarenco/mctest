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
        'tokens': tokenize(row['story']),
        'questions': [{
            'tokens': tokenize(question_text(row['q%d' % q_number])),
            'answers': [tokenize(row['a%d%d' % (q_number, a_number)])
                        for a_number in xrange(1, 5)],
            'type': question_type(row['q%d' % q_number])
        } for q_number in xrange(1, 5)]
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Converts raw TSV files from the MCTest dataset')
    _arg = parser.add_argument
    _arg('-o', type=str, action='store', metavar='FORMAT',
         default=DEFAULT_OUTPUT_FORMAT,
         help='Output format: json (default=%s)' %  DEFAULT_OUTPUT_FORMAT)
    _arg('--rm-stop', type=str, action='store', metavar='FILE',
         help='Remove stop words specified by file (one word per line).')
    _arg('--rm-punct', action='store_true',
         help='Remove punctuation when tokenizing.')
    _arg('-i', type=str, action='store', metavar='FILE', default=None,
         help='Input file (TSV).')
    args = parser.parse_args()

    token_filters = []
    if args.rm_stop:
        stopwords = open(args.rm_stop, 'r').read().split('\n')
        stopwords = set(map(lambda x: x.strip().rstrip(), stopwords))
        token_filters.append(lambda x: x.lower() not in stopwords)
    if args.rm_punct:
        token_filters.append(lambda x: x not in PUNCTS)

    def tokenize(text):
        if not isinstance(text, basestring):
            text = str(text)
        text = text.replace('\\newline', ' ')
        return filter(lambda t: all(map(lambda f: f(t), token_filters)),
                      nltk.word_tokenize(text))

    with (open(args.i, 'r') if args.i else sys.stdin) as fin:
        df = pd.read_csv(fin, sep='\t', names=COLUMNS)
        if args.o == 'json':
            try:
                for row in (df.ix[i] for i in df.index):
                    datapoint = row_to_dict(row, tokenize)
                    print(json.dumps(datapoint), file=sys.stdout)
            except IOError as e:
                if e.errno == errno.EPIPE:
                    sys.exit(0)
                raise e
        else:
            print('Unknown output format "%s"' % args.o, file=sys.stderr)
            sys.exit(2)
