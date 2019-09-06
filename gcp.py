#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from goplus.parser import Parser
from goplus.tokenizer import Tokenizer

FNAME = '/Users/chenzhuoyu/GolangProjects/pkg/mod/github.com/nyaruka/phonenumbers@v1.0.45/prefix_to_geocodings_bin.go'

def main():
    with open(FNAME, 'r', newline = None) as fp:
        Parser(Tokenizer(fp.read(), FNAME)).parse()

if __name__ == '__main__':
    main()
