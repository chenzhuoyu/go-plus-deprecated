#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import unittest

from goplus.inferrer import Inferrer

GOROOT = os.environ.get('GOROOT', '')
GOPATH = os.environ.get('GOPATH', '').split(os.path.pathsep)

GOPKG = os.environ.get('GOPKG', '')
GOPROJ = os.path.join(GOPATH[0], 'src', GOPKG)

class TestInferrer(unittest.TestCase):
    def test_inferrer(self):
        print('GOROOT :', GOROOT)
        print('GOPATH :', GOPATH)
        print('GOPKG  :', GOPKG)
        print('GOPROJ :', GOPROJ)
        ifr = Inferrer('darwin', 'amd64', GOPROJ, GOROOT, GOPATH)
        ifr.infer(GOPKG)

if __name__ == '__main__':
    unittest.main()
