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


def answer_distance(passage, question, answer):
    if not isinstance(question, set):
        question = set(question)
    if not isinstance(answer, set):
        answer = set(answer)
    s_question = question.intersection(passage)
    s_answer = answer.intersection(passage).difference(question)
    if len(s_question) == 0 or len(s_answer) == 0:
        return 1.0
    last_q, last_a = np.inf, np.inf
    closest = np.inf
    for i, token in enumerate(passage):
        if token in s_question:
            last_q = i
        if token in s_answer:
            last_a = i
        if abs(last_q - last_a) < closest:
            # print(last_q, last_a)
            closest = (np.abs(last_q - last_a) + 1) / len(passage)
    return closest


class SlidingWindow(object):
    def __init__(self):
        pass

    def fit(self, passages, window_size=None):
        self._icounts = compute_inverse_counts(passages)
        self._window_size = window_size

    def predict_target(self, tokens, target, verbose=True):
        if not isinstance(target, set):
            target = set(target)
        target_size = len(target)
        max_overlap_score = 0.0
        tokens_at_max = []
        for i in xrange(len(tokens)):
            overlap_score = 0.0
            try:
                window_size = self._window_size or target_size
                for j in xrange(window_size):
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

    def predict(self, story, verbose=True):
        p_tokens = story['tokens']
        answers = []
        for question in story['questions']:
            scores = []
            q_tokens = question['tokens']
            if verbose:
                print('Question: %s' % q_tokens)
            for answer in question['answers']:
                dist = answer_distance(p_tokens, q_tokens, answer)
                scores.append(self.predict_target(
                    p_tokens, set(q_tokens + answer), verbose) - dist)
            answers.append(scores)
        return answers


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Baseline models from the MCTest paper (sliding '
        'window and distance based)')
    _arg = parser.add_argument
    _arg('--train', type=str, action='store', metavar='FILE', required=True,
         help='File with stories and questions (JSON format).')
    _arg('--truth', type=str, action='store', metavar='FILE',
         help='File with correct answers to the questions.')
    args = parser.parse_args()

    stories = []
    with open(args.train, 'r') as train_in:
        stories.extend(map(json.loads, train_in.readlines()))

    for ws in xrange(5, 30):
        sw = SlidingWindow()
        sw.fit(stories, window_size=None)
        predicted = []
        for story in stories[:]:
            for a in sw.predict(story, False):
                predicted.append(ANSWER_LETTER[a.index(max(a))])

        if args.truth:
            answers_in = open(args.truth, 'r')
            answers = np.array(load_target_answers(answers_in))
            predicted = np.array(predicted)
            assert len(answers) == len(predicted)
            print('[ws=%d] Accuracy: %.4f' %
                  (ws, np.sum(answers == predicted) / float(len(predicted))))
        break
    # target = set( + q['questions'][0]['answers'][1])
    # sliding_window_overlap(icounts, questions[0]['passage'], target)
