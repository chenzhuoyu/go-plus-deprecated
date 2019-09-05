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

from goplus.modules import Module
from goplus.modules import Reader
from goplus.modules import Resolver

def _do_parse(fname: str) -> Package:
    with open(fname, newline = None) as fp:
        return Parser(Tokenizer(fp.read(), fname)).parse()

def _make_string(value: str) -> String:
    return String(Token(0, 0, '<main>', TokenType.String, value.encode('utf-8')))

def _parse_modules() -> Module:
    with open(os.path.join(os.environ.get('TEST_GO_PROJ', '.'), 'go.mod'), newline = None) as fp:
        return Reader().parse(fp.read())

class TestInferrer(unittest.TestCase):
    def test_inferrer(self):
        ifr = Inferrer('amd64', 'darwin', False, _do_parse, Resolver(os.environ.get('TEST_GO_PATH', '').split(':')))
        ifr.infer_package(_make_string(os.environ.get('TEST_GO_PKG', '')), None)

if __name__ == '__main__':
    unittest.main()
