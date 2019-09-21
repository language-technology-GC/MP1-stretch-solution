#!/usr/bin/env python
"""Computes raw unigram and co-occurrence statistics.

This script takes a tokenized (and case-folded, etc. if desired) corpus and
produces two TSV files:

* a unigram frequency table
* a co-occurrence frequency table, where the pairs are in lexicographic order

The user specifies the input and output paths and the token window.
"""

import argparse
import collections
import logging


M = 1_000_000  # One million.


def main(args: argparse.Namespace) -> None:
    # Collects frequencies.
    unigram_freqs = collections.Counter()
    cooccur_freqs = collections.Counter()
    with open(args.tok_path) as source:
        for (linenum, line) in enumerate(source, 1):
            sentence = line.rstrip().split()
            for (i, target_token) in enumerate(sentence):
                unigram_freqs[target_token] += 1
                window = (
                    sentence[max(i - args.ws, 0):i]
                    + sentence[i + 1:i + args.ws]
                )
                for context_token in window:
                    # Prevents "symmetrical" or double counts.
                    if target_token > context_token:
                        continue
                    key = (target_token, context_token)
                    cooccur_freqs[key] += 1
            if linenum % M == 0:
                logging.info("%dm lines processed", linenum // M)
    # Writes them out.
    logging.info("Writing unigrams...")
    with open(args.unigram_path, "w") as sink:
        for (token, freq) in unigram_freqs.most_common():
            print(f"{token}\t{freq}", file=sink)
    logging.info("Writing co-occurrences...")
    # For reasons I don't understand, the sorting of items implicit in
    # most_common requires an enormous amount of memory, so we don't bother.
    with open(args.cooccur_path, "w") as sink:
        for ((target_token, context_token), freq) in cooccur_freqs.items():
            print(f"{target_token} {context_token}\t{freq}", file=sink)


if __name__ == "__main__":
    logging.basicConfig(format="%(levelname)s: %(message)s", level="INFO")
    parser = argparse.ArgumentParser(description="Co-occurrence calculator")
    parser.add_argument("tok_path", help="input tokenized text")
    parser.add_argument("unigram_path", help="output unigram TSV")
    parser.add_argument("cooccur_path", help="output co-occurrence TSV")
    parser.add_argument(
        "--ws",
        default=5,
        help="window size for co-occurrence (default: %(default)s)",
    )
    main(parser.parse_args())
