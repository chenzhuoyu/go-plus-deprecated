#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from goplus.tokenizer import Tokenizer

FNAME = '/Users/chenzhuoyu/GolangProjects/IDL/ci/vendor/github.com/fatih/color/color.go'

def main():
    with open(FNAME, 'r', newline = None) as fp:
        tk = Tokenizer(fp.read(), FNAME)
    while not tk.is_eof:
        print(tk.next())

if __name__ == '__main__':
    main()
