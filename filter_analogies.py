#!/usr/bin/python3

import self_assess
import argparse

parser = argparse.ArgumentParser(description='Filter out analogies from input file based on words in model.')
parser.add_argument('infile', type=str, help='Analogies input file')
parser.add_argument('outfile', type=str, help='Filtered output file')
args = parser.parse_args()

_, words, _ = self_assess.extract_words(5000)

word_set = set(words)

with open(args.infile, 'r') as f:
    with open(args.outfile, 'w') as out:
        header = None
        for line in f:
            l = line.split()
            if len(l) != 4:
                # section header
                header = line
            else:
                if l[0] in word_set and l[1] in word_set and l[2] in word_set and l[3] in word_set:
                    if header is not None:
                        out.write(header)
                        header = None
                    out.write(line)

