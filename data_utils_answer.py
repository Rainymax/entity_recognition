""" Named entity recognition fine-tuning: utilities to work with CoNLL-2003 task. """

from __future__ import absolute_import, division, print_function

import logging
import os
from io import open
from typing import *

logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for token classification."""

    def __init__(self, guid, words, labels):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            words: list. The words of the sequence.
            labels: (Optional) list. The labels for each word of the sequence. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.words = words
        self.labels = labels
        assert len(words) == len(labels)
    
    def __len__(self):
        return len(self.words)


def read_examples_from_file(file_path, mode):
    """
    Read file and load into a list of `InputExample`s

    Args:
        file_path: str, file path to load
        mode: str, "train" or "test"
    Returns:
        examples = List[InputExample]
    """
    guid_index = 1
    examples = []
    with open(file_path, encoding="utf-8") as f:
        words = []
        labels = []
        for line in f:
            if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                if words:
                    examples.append(InputExample(guid="{}-{}".format(mode, guid_index),
                                                 words=words,
                                                 labels=labels))
                    guid_index += 1
                    words = []
                    labels = []
            else:
                splits = line.split(" ")
                words.append(splits[0])
                if len(splits) > 1:
                    labels.append(splits[-1].replace("\n", ""))
                else:
                    # Examples could have no label for mode = "test"
                    labels.append("O")
        if words:
            examples.append(InputExample(guid="%s-%d".format(mode, guid_index),
                                         words=words,
                                         labels=labels))
    return examples

def word2features(sent, i):
    """
    get discrete features for a single word
    Args:
        sent: InputExample
        i: int, index for target word
    Returns:
        features: List[str]
    
    Please design features for one word in the sentence as input into pycrfsuite model (https://python-crfsuite.readthedocs.io/en/latest/)
    """
    word = sent.words[i]
    features = [
        'bias',
        'word.lower=' + word.lower(),
        'word.alpha=%s' % word.isalpha(),
        'word.isupper=%s' % word.isupper(),
        'word.isdigit=%s' % word.isdigit(),
    ]
    if i > 0:
        word1 = sent.words[i-1]
        features.extend([
            '-1:word.lower=' + word1.lower(),
            '-1:word.alpha=%s' % word.isalpha(),
            '-1:word.isupper=%s' % word1.isupper(),
            '-1:word.isdigit=%s' % word.isdigit(),
        ])
    else:
        features.append('BOS')
        
    if i < len(sent)-1:
        word1 = sent.words[i+1]
        features.extend([
            '+1:word.lower=' + word1.lower(),
            '-1:word.alpha=%s' % word.isalpha(),
            '+1:word.isupper=%s' % word1.isupper(),
            '+1:word.isdigit=%s' % word.isdigit(),
        ])
    else:
        features.append('EOS')
                
    return features


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    return sent.labels

def sent2tokens(sent):
    return sent.words
