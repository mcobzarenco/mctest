#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function, division
import argparse
import json
import os
import struct
import sys
from base64 import urlsafe_b64decode as b64decode, \
    urlsafe_b64encode as b64encode
from collections import defaultdict
from functools import partial
from itertools import imap

import numpy as np

from mctest_pb2 import StoryAsWords, QuestionAsWords
from parse import parse_proto_stream


ANSWER_LETTER = ['A', 'B', 'C', 'D']


def compute_counts(stories):
    counts = defaultdict(lambda: 0.0)
    for story in stories:
        for token in story.passage:
            counts[token] += 1.0
    return counts


def compute_inverse_counts(stories):
    counts = compute_counts(stories)
    icounts = {}
    for token, token_count in counts.iteritems():
        icounts[token] = np.log(1.0 + 1.0 / token_count)
    return icounts


def baseline_distance(passage, question, answer):
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
            closest = np.abs(last_q - last_a) / (len(passage) - 1)
    assert closest > 0 and closest <= 1
    return closest


class SlidingWindow(object):
    def __init__(self):
        pass

    def fit(self, stories, window_size=None):
        self._icounts = compute_inverse_counts(stories)
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
                tokens_at_max = tokens[i:i + window_size]
                max_overlap_score = overlap_score
        if verbose:
            print('[score=%.2f for target=%s] passage: %s ' %
                  (max_overlap_score, target, tokens_at_max), file=sys.stderr)
        return max_overlap_score

    def predict(self, passage, question, answers,
                with_distance=True, verbose=True):
        scores = []
        if verbose:
            print('Question: %s' % question)
        for answer in answers:
            dist = baseline_distance(passage, question, answer) \
                   if with_distance else 0
            scores.append(self.predict_target(
                passage, set(question + answer), verbose) - dist)
        return scores


def load_target_answers(stream):
    answers = stream.readlines()
    answers = map(lambda x: x.rstrip().split('\t'), answers)
    return reduce(lambda x, y: x + y, answers)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Baseline models from the MCTest paper (sliding '
        'window and distance based)')
    _arg = parser.add_argument
    _arg('--train', type=str, action='store', metavar='FILE', required=True,
         help='File with stories and questions (JSON format).')
    _arg('--truth', type=str, action='store', metavar='FILE',
         help='File with correct answers to the questions.')
    _arg('--window-size', type=int, action='store', metavar='SIZE',
         default=None, help='Fixed window size for the sliding window ' \
         'algorithm. By default it has the same length as the question.')
    _arg('--distance', action='store_true',
         help='Substract the baseline distance measure.')
    args = parser.parse_args()

    stories = list(parse_proto_stream(open(args.train, 'r')))
    print('[model]\nwindow_size = %s\ndistance = %s\n' %
          (args.window_size, args.distance))

    sw = SlidingWindow()
    sw.fit(stories, window_size=args.window_size)
    predicted, q_types = [], []
    for story in stories:
        passage = story.passage
        for question in story.questions:
            q_types.append(question.type)
            answer_tokens = map(lambda x: list(x.tokens), question.answers)
            scores = sw.predict(passage, list(question.tokens), answer_tokens,
                                with_distance=args.distance, verbose=False)
            predicted_letter = ANSWER_LETTER[scores.index(max(scores))]
            # print('scores: %s (%s)' % (scores, predicted_letter))
            predicted.append(predicted_letter)

    if args.truth:
        answers_in = open(args.truth, 'r')
        answers = np.array(load_target_answers(answers_in))
        predicted = np.array(predicted)
        assert len(answers) == len(predicted)

        single = np.array(q_types) == QuestionAsWords.ONE
        n_single = float(np.sum(single))
        n_multiple = float(np.sum(~single))
        assert n_single + n_multiple == len(answers)

        print('[results]')
        print('All accuracy [%d]: %.4f' %
              (n_single + n_multiple,
               np.sum(answers == predicted) / float(len(predicted))))
        print('Single accuracy [%d]: %.4f' %
              (n_single,
               np.sum(answers[single] == predicted[single]) / n_single))
        print('Multiple accuracy [%d]: %.4f' %
              (n_multiple,
               np.sum(answers[~single] == predicted[~single]) / n_multiple))
    else:
        for p in predicted:
            print(p, file=sys.stdout)
