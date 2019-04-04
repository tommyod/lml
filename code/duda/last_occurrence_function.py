#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 12:10:52 2018

@author: tommy
"""

import string
import collections


def last_occurence(alphabet, word):
    """
    Returns a mapping F[char] -> last occurrence in word.
    """
    mapping = {i: -1 for i in alphabet}  # This is O(d)

    for index, char in enumerate(word):  # This is O(m)
        mapping[char] = index

    return mapping


def last_occurence_defaultdict(alphabet, word):
    """
    Returns a mapping F[char] -> last occurrence in word.
    """
    mapping = collections.defaultdict(lambda: -1)  # This is O(1)

    for index, char in enumerate(word):  # This is O(m)
        mapping[char] = index

    return mapping


alphabet = string.ascii_lowercase
word = "abcdabdabcaabcda"
print(last_occurence(alphabet, word))
print(last_occurence_defaultdict(alphabet, word))
