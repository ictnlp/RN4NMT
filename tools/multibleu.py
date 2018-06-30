#!/usr/bin/env python
# Ander Martinez Sanchez

from __future__ import division, print_function
from math import exp, log
from collections import Counter
from tools.utils import *
from tools.bleu import zh_to_chars

def ngram_count(words, n):
    if n <= len(words):
        return Counter(zip(*[words[i:] for i in range(n)]))
    return Counter()


def max_count(c1, c2):
    return Counter({k: max(c1[k], c2[k]) for k in c1})


def min_count(c1, c2):
    return Counter({k: min(c1[k], c2[k]) for k in c1})


def closest_min_length(candidate, references):
    l0 = len(candidate)
    return min((abs(len(r) - l0), len(r)) for r in references)[1]


def safe_log(n):
    if n <= 0:
        return -9999999999
    return log(n)


def precision_n(candidate, references, n):
    ref_max = reduce(max_count, [ngram_count(ref, n) for ref in references])
    candidate_ngram_count = ngram_count(candidate, n)
    total = sum(candidate_ngram_count.values())
    correct = sum(reduce(min_count, (ref_max, candidate_ngram_count)).values())
    score = (correct / total) if total else 0
    return score, correct, total


def bleu(candidate, references, maxn=4):
    precs = [precision_n(candidate, references, n) for n in range(1, maxn+1)]
    bp = exp(1 - closest_min_length(candidate, references) / len(candidate))
    return bp * exp(sum(safe_log(precs[n]) for n in range(maxn)) / maxn)


def tokenize(txt, char=False):
    txt = txt.strip()
    if char is True: txt = zh_to_chars(txt)
    else: txt = txt.split()
    return txt

def tokenize_lower(txt, char=False):
    txt = txt.strip().lower()
    if char is True: txt = zh_to_chars(txt)
    else: txt = txt.split()
    return txt

def multi_bleu(candidates, all_references, tokenize_fn=tokenize, maxn=4, char=False):
    correct = [0] * maxn
    total = [0] * maxn
    cand_tot_length = 0
    ref_closest_length = 0

    for candidate, references in zip(candidates, zip(*all_references)):
        candidate = tokenize_fn(candidate, char)
        references = map(tokenize_fn, references, [char for _ in references])
        cand_tot_length += len(candidate)
        ref_closest_length += closest_min_length(candidate, references)
        for n in range(maxn):
            sc, cor, tot = precision_n(candidate, references, n + 1)
            correct[n] += cor
            total[n] += tot

    precisions = [(correct[n] / total[n]) if correct[n] else 0 for n in range(maxn)]

    if cand_tot_length < ref_closest_length:
        brevity_penalty = exp(1 - ref_closest_length / cand_tot_length)
    else:
        brevity_penalty = 1
    score = 100 * brevity_penalty * exp(
                    sum(safe_log(precisions[n]) for n in range(maxn)) / maxn)
    prec_pc = [100 * p for p in precisions]
    return score, prec_pc, brevity_penalty, cand_tot_length, ref_closest_length


def print_multi_bleu(cand_file, ref_fpaths, cased=False, maxn=4, char=False):

    tokenize_fn = tokenize if cased is True else tokenize_lower
    if cased is False: wlog('Calculating case-insensitive {}-gram BLEU ...'.format(maxn))
    else: wlog('Calculating case-sensitive {}-gram BLEU ...'.format(maxn))
    wlog('\tcandidate file: {}'.format(cand_file))
    wlog('\treferences file:')
    for ref in ref_fpaths: wlog('\t\t{}'.format(ref))

    cand = open(cand_file, 'r')
    refs = [open(ref_fpath, 'r') for ref_fpath in ref_fpaths]

    score, precisions, brevity_penalty, cand_tot_length, ref_closest_length = \
        multi_bleu(cand, refs, tokenize_fn, maxn, char=char)

    cand.close()
    for fd in refs: fd.close()

    wlog('BLEU = {:.2f}, {:.1f}/{:.1f}/{:.1f}/{:.1f} '
         '(BP={:.3f}, ratio={:.3f}, hyp_len={:d}, ref_len={:d})'.format(
            score, precisions[0], precisions[1], precisions[2], precisions[3],
            brevity_penalty, cand_tot_length / ref_closest_length, cand_tot_length,
            ref_closest_length))

    score = float('%.2f' % (score))
    return score

if __name__ == "__main__":
    import os
    import sys
    import argparse

    parser = argparse.ArgumentParser(description='BLEU score on multiple references.')
    parser.add_argument('-lc', help='Lowercase', action='store_true')
    parser.add_argument('-c', help='translation file')
    parser.add_argument('reference', help='Reads the reference_[0, 1, ...]')
    args = parser.parse_args()
    tokenize_fn = tokenize_lower if args.lc else tokenize

    ref_fpaths = []
    ref_cnt = 2
    if ref_cnt == 1:
        ref_fpath = args.reference
        if os.path.exists(ref_fpath): ref_fpaths.append(ref_fpath)
    else:
        for idx in range(ref_cnt):
            ref_fpath = '{}_{}'.format(args.reference, idx)
            if not os.path.exists(ref_fpath): continue
            ref_fpaths.append(ref_fpath)

    #print(args.reference)
    # TODO: Multiple references
    #reference_files = [args.reference]
    print(ref_fpaths)

    #open_files = map(open, ref_fpaths)
    cand_file = args.c
    cased = ( not args.lc )
    print_multi_bleu(cand_file, ref_fpaths, cased, 4)
