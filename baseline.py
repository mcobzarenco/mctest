#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function, division
import argparse
import json
import os
import sys
from base64 import urlsafe_b64decode as b64decode, \
    urlsafe_b64encode as b64encode
from collections import defaultdict
from functools import partial
from itertools import imap

import numpy as np


ANSWER_LETTER = ['A', 'B', 'C', 'D']


def compute_counts(passages):
    counts = defaultdict(lambda: 0.0)
    for passage in passages:
        for token in passage['tokens']:
            counts[token] += 1.0
    return counts


def compute_inverse_counts(passages):
    counts = compute_counts(passages)
    icounts = {}
    for (token, token_count) in counts.iteritems():
        icounts[token] = np.log(1.0 + 1.0 / token_count)
    return icounts


def load_target_answers(stream):
    answers = stream.readlines()
    answers = map(lambda x: x.rstrip().split('\t'), answers)
    return reduce(lambda x, y: x + y, answers)


class SlidingWindow(object):
    def __init__(self):
        pass

    def fit(self, passages):
        self._icounts = compute_inverse_counts(passages)

    def predict_target(self, tokens, target, verbose=True):
        if not isinstance(target, set):
            target = set(target)
        target_size = len(target)
        max_overlap_score = 0.0
        tokens_at_max = []
        for i in xrange(len(tokens)):
            overlap_score = 0.0
            try:
                for j in xrange(target_size):
                    if tokens[i + j] in target:
                        overlap_score += self._icounts[tokens[i + j]]
            except IndexError:
                pass
            if overlap_score > max_overlap_score:
                tokens_at_max = tokens[i:i + target_size]
                max_overlap_score = overlap_score
        if verbose:
            print('[score=%.2f for target=%s] passage: %s ' %
                  (max_overlap_score, target, tokens_at_max), file=sys.stderr)
        return max_overlap_score

    def predict(self, passage, verbose=True):
        p_tokens = passage['tokens']
        answers = []
        for question in passage['questions']:
            scores = []
            q_tokens = question['tokens']
            if verbose:
                print('Question: %s' % q_tokens)
            for answer in question['answers']:
                scores.append(self.predict_target(
                    p_tokens, set(q_tokens + answer), verbose))
            answers.append(scores)
        return answers


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Baseline models from the MCTest paper (sliding '
        'window and distance based)')
    _arg = parser.add_argument
    _arg('--train', type=str, action='store', metavar='FILE', required=True,
         help='File with passages and questions (JSON format).')
    _arg('--truth', type=str, action='store', metavar='FILE',
         help='File with correct answers to the questions.')
    args = parser.parse_args()

    passages = []
    with open(args.train, 'r') as train_in:
        passages.extend(map(json.loads, train_in.readlines()))

    sw = SlidingWindow()
    sw.fit(passages)
    predicted = []
    for passage in passages:
        for a in sw.predict(passage, False):
            predicted.append(ANSWER_LETTER[a.index(max(a))])

    if args.truth:
        answers_in = open(args.truth, 'r')
        answers = np.array(load_target_answers(answers_in))
        predicted = np.array(predicted)
        assert len(answers) == len(predicted)
        print('Accuracy: %.4f' % (np.sum(answers == predicted) / float(len(predicted))))

    # target = set( + q['questions'][0]['answers'][1])
    # sliding_window_overlap(icounts, questions[0]['passage'], target)
