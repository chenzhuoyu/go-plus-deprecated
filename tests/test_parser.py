import unittest

from goplus.parser import Parser
from goplus.tokenizer import Tokenizer

_import_src = """
package test

import   "lib/math"         // math.Sin
import m "lib/math"         // m.Sin
import . "lib/math"         // Sin
import (
    `context`
    `fmt`

    `example.com/gopkg/gorm`
    `github.com/sirupsen/logrus`

    `example.com/ee/people/saas/minicorehr/biz/dal`
    `example.com/ee/people/saas/minicorehr/biz/models`
    _ `example.com/ee/people/saas/minicorehr/biz/models/db`
)
"""

_const_src = """
package test

const Pi float64 = 3.14159265358979323846
const zero = 0.0         // untyped floating-point constant
const (
    size int64 = 1024
    eof        = -1  // untyped integer constant
)
const a, b, c = 3, 4, "foo"  // a = 3, b = 4, c = "foo", untyped integer and string constants
const u, v float32 = 0, 3    // u = 0.0, v = 3.0

const (
    Sunday = iota
    Monday
    Tuesday
    Wednesday
    Thursday
    Friday
    Partyday
    numberOfDays  // this constant is not exported
)

const (
    c0 = iota  // c0 == 0
    c1 = iota  // c1 == 1
    c2 = iota  // c2 == 2
)

const (
    a = 1 << iota  // a == 1  (iota == 0)
    b = 1 << iota  // b == 2  (iota == 1)
    c = 3          // c == 3  (iota == 2, unused)
    d = 1 << iota  // d == 8  (iota == 3)
)

const (
    u         = iota * 42  // u == 0     (untyped integer constant)
    v float64 = iota * 42  // v == 42.0  (float64 constant)
    w         = iota * 42  // w == 84    (untyped integer constant)
)

const x = iota  // x == 0
const y = iota  // y == 0

const (
    bit0, mask0 = 1 << iota, 1<<iota - 1  // bit0 == 1, mask0 == 0  (iota == 0)
    bit1, mask1                           // bit1 == 2, mask1 == 1  (iota == 1)
    _, _                                  //                        (iota == 2, unused)
    bit3, mask3                           // bit3 == 8, mask3 == 7  (iota == 3)
)
"""

class TestParser(unittest.TestCase):
    def test_import(self):
        Parser(Tokenizer(_import_src, 'test.go')).parse()

    def test_const(self):
        print(Parser(Tokenizer(_const_src, 'test.go')).parse())

if __name__ == '__main__':
    unittest.main()
