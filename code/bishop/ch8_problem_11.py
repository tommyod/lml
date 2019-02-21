#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 21:00:24 2019

@author: tommy
"""

import itertools


def probDgivenG(D, G):
    if G == 1:
        if D == 1:
            return 0.9
        return 1 - probDgivenG(not D, G)

    if G == 0:
        if D == 0:
            return 0.9
        return 1 - probDgivenG(not D, G)


def probGgivenBF(G, B, F):
    results = {(0, 0): 0.1, (0, 1): 0.2, (1, 0): 0.2, (1, 1): 0.8}

    if G == 1:
        return results[(B, F)]
    else:
        return 1 - probGgivenBF(not G, B, F)


def probB(B):
    if B == 1:
        return 0.9
    return 1 - probB(not B)


def probF(F):
    if F == 1:
        return 0.9
    return 1 - probF(not F)


# Evaluating P(D = 0)
prob_D_is_zero = 0
for G, F, B in itertools.product([0, 1], repeat=3):

    prob_D_is_zero += (
        probDgivenG(D=0, G=G) * probGgivenBF(G, B, F) * probB(B) * probF(F)
    )

print("p(D=0) =", prob_D_is_zero)


# Evaluating P(D = 0 | F = 0)
ans = 0
for G, B in itertools.product([0, 1], repeat=2):

    ans += probDgivenG(D=0, G=G) * probGgivenBF(G, B, F=0) * probB(B)


print("p(D=0 | F=0) =", ans)

# -----------------------------------------------------------------------

# Evaluating P(D = 0 , B =0)
ans = 0
for G, F in itertools.product([0, 1], repeat=2):

    ans += probDgivenG(D=0, G=G) * probGgivenBF(G, B=0, F=F) * probB(B=0) * probF(F)


print("p(D=0, B=0) =", ans)


# Evaluating P(D = 0 , B =0, F=0)
ans = 0
for (G,) in itertools.product([0, 1], repeat=1):

    ans += probDgivenG(D=0, G=G) * probGgivenBF(G, B=0, F=0) * probB(B=0) * probF(F=0)


print("P(D=0 , B=0, F=0) =", ans)
