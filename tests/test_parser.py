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
    _ `unsafe`
)
"""

_const_src = """
package test

type Foo struct {
    x, y      float64 ""  // an empty tag string is like an absent tag
    name, val string  "any string is permitted as a tag"
    _         []byte  "ceci n'est pas un champ de structure"
    _         int
}

// A struct corresponding to a TimeStamp protocol buffer.
// The tag strings define the protocol buffer field numbers;
// they follow the convention outlined by the reflect package.
type Bar struct {
    microsec  uint64 `protobuf:"1"`
    serverIP6 uint64 `protobuf:"2"`
}

type (
    nodeList = []*Node  // nodeList and []*Node are identical types
    Polar    = polar    // Polar and polar denote identical types
)

type (
    Point struct{ x, y float64 }  // Point and struct{ x, y float64 } are different types
    polar Point                   // polar and Point denote different types
)

type TreeNode struct {
    left, right *TreeNode
    value *Comparable
}

# type Block interface {
#     BlockSize() int
#     Encrypt(src, dst []byte)
#     Decrypt(src, dst []byte)
# }

// A Mutex is a data type with two methods, Lock and Unlock.
type Mutex struct         { /* Mutex fields */ }
# func (m *Mutex) Lock()    { /* Lock implementation */ }
# func (m *Mutex) Unlock()  { /* Unlock implementation */ }

// NewMutex has the same composition as Mutex but its method set is empty.
type NewMutex Mutex

// The method set of PtrMutex's underlying type *Mutex remains unchanged,
// but the method set of PtrMutex is empty.
type PtrMutex *Mutex

// The method set of *PrintableMutex contains the methods
// Lock and Unlock bound to its embedded field Mutex.
type PrintableMutex struct {
    Mutex
}

// MyBlock is an interface type that has the same method set as Block.
type MyBlock Block

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

# const (
#     bit0, mask0 = 1 << iota, 1<<iota - 1  // bit0 == 1, mask0 == 0  (iota == 0)
#     bit1, mask1                           // bit1 == 2, mask1 == 1  (iota == 1)
#     _, _                                  //                        (iota == 2, unused)
#     bit3, mask3                           // bit3 == 8, mask3 == 7  (iota == 3)
# )

var i int
var U, V, W float64
var k = 0
var x, y float32 = -1, -2
var (
    i2      int
    u, v, s = 2.0, 3.0, "bar"
)
var re, im = complexSqrt(-1)
var _, found = entries[name]  // map lookup; only interested in "found"

var d = math.Sin(0.5)  // d is float64
var i3 = 42            // i is int
var t, ok = x.(T)      // t is T, ok is bool
var n = nil            // illegal
"""

class TestParser(unittest.TestCase):
    def test_import(self):
        Parser(Tokenizer(_import_src, 'test.go')).parse()

    def test_const(self):
        print(Parser(Tokenizer(_const_src, 'test.go')).parse())

if __name__ == '__main__':
    unittest.main()
