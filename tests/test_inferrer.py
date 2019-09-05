#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import unittest

from goplus.ast import String
from goplus.ast import Package

from goplus.parser import Parser
from goplus.inferrer import Inferrer
from goplus.tokenizer import Tokenizer

from goplus.tokenizer import Token
from goplus.tokenizer import TokenType

def _do_parse(fname: str) -> Package:
    with open(fname, newline = None) as fp:
        return Parser(Tokenizer(fp.read(), fname)).parse()

def _make_string(value: str) -> String:
    return String(Token(0, 0, '<main>', TokenType.String, value.encode('utf-8')))

class TestInferrer(unittest.TestCase):
    def test_inferrer(self):
        ifr = Inferrer(_do_parse, os.environ.get('TEST_GO_PATH', '').split(':'))
        ifr.infer_package(_make_string(os.environ.get('TEST_GO_PKG', '')))

if __name__ == '__main__':
    unittest.main()
