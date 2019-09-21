#!/usr/bin/env python
"""Pointwise mutual information calculation."""


import argparse
import logging
import csv
import math

import scipy.stats


from typing import Dict, List, Tuple


Pair = Tuple[str, str]
Pairs = List[Pair]

Unigram = Dict[str, int]
Cooccur = Dict[Pair, int]


def _make_key(w1: str, w2: str) -> Pair:
    """Sorts the key."""
    return (w1, w2) if w1 < w2 else (w2, w1)


def _read_unigram(unigram_path: str) -> Unigram:
    """Reads unigram frequency table."""
    table = {}
    with open(unigram_path, "r") as source:
        for line in source:
            (token, _freq) = line.rstrip().split("\t", 1)
            table[token] = int(_freq)
    return table


def _read_cooccur(cooccur_path: str) -> Cooccur:
    """Reads cooccurrence frequency table."""
    table = {}
    with open(cooccur_path, "r") as source:
        for line in source:
            (pair, _freq) = line.rstrip().split("\t", 1)
            (w1, w2) = tuple(pair.split(" ", 1))
            key = _make_key(w1, w2)
            freq = int(_freq)
            table[key] = freq
    return table


def _cor(x, y) -> float:
    """Computes Spearman correlation coefficient."""
    return scipy.stats.spearmanr(x, y).correlation


class PMICalculator:
    """Computes pointwise mutual information statistics."""

    def __init__(self, unigram: Unigram, cooccur: Cooccur):
        self.unigram = unigram
        self.cooccur = cooccur
        self.unigram_n = sum(self.unigram.values())
        self.cooccur_n = sum(self.cooccur.values())

    def _unigram_p(self, w: str) -> float:
        return self.unigram.get(w, 0) / self.unigram_n

    def _cooccur_p(self, w1: str, w2: str) -> float:
        key = _make_key(w1, w2)
        return self.cooccur.get(key, 0) / self.cooccur_n

    # TODO: There are many clever opportunities for log-math here but which
    # if any should I take? It is unclear.

    def pmi(self, w1: str, w2: str) -> float:
        """Computes pointwise mutual information."""
        numerator = self._cooccur_p(w1, w2)
        if not numerator:
            return float("-inf")
        denominator = self._unigram_p(w1) * self._unigram_p(w2)
        return math.log2(numerator / denominator)

    def ppmi(self, w1: str, w2: str) -> float:
        """Computes positive pointwise mutual information."""
        return max(self.pmi(w1, w2), 0.0)


def main(args: argparse.Namespace) -> None:
    logging.info("Reading human similarity scores from %s", args.table_path)
    pairs: Pairs = []
    human_scores: List[float] = []
    with open(args.table_path, "r") as source:
        for row in csv.DictReader(source, delimiter="\t"):
            w1 = row["Word 1"].casefold()
            w2 = row["Word 2"].casefold()
            # Gives them a canonical order.
            key = _make_key(w1, w2)
            pairs.append(key)
            human_scores.append(float(row["Human (mean)"]))
    # Sets up PMI calculations.
    logging.info("Reading unigram counts from %s", args.unigram_path)
    unigram = _read_unigram(args.unigram_path)
    logging.info("Reading co-occurrence counts from %s", args.cooccur_path)
    cooccur = _read_cooccur(args.cooccur_path)
    logging.info("Computing PPMI")
    calc = PMICalculator(unigram, cooccur)
    ppmi_scores: List[float] = []
    for (w1, w2) in pairs:
        ppmi_scores.append(calc.ppmi(w1, w2))
    logging.info("PPMI:\t%.4f", _cor(human_scores, ppmi_scores))


if __name__ == "__main__":
    logging.basicConfig(format="%(levelname)s: %(message)s", level="INFO")
    parser = argparse.ArgumentParser(description="PMI calculation")
    parser.add_argument("unigram_path")
    parser.add_argument("cooccur_path")
    parser.add_argument("table_path")
    main(parser.parse_args())
