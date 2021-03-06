#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest

from goplus.parser import Parser
from goplus.tokenizer import Tokenizer

_import_src = r"""package test

import   "lib/math"         // math.Sin
import m "lib/math"         // m.Sin
import . "lib/math"         // Sin

import (
    `context`
    _ `unsafe`
)
"""

_const_src = r"""package test

// #include <stdio.h>
// #include <stdlib.h>
import `C`

/*
#include <stdint.h>
*/
import `C`

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

type Block interface {
    EmbeddedInterface
    BlockSize() int
    Encrypt(src, dst []byte)
    Decrypt(src, dst []byte)
    Concat(a interface{}, args ...interface{})
    MethodWithOnlyTypes(int, int)
    MethodWithMultipleReturns(int, int) (int, error)
    MethodWithMultipleNamedReturns(int, int) (a int, b error)
}

// A Mutex is a data type with two methods, Lock and Unlock.
type Mutex struct         { /* Mutex fields */ }
func (m *Mutex) Lock()    { /* Lock implementation */ }
func (m *Mutex) Unlock()  { /* Unlock implementation */ }

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

const (
    bit0, mask0 = 1 << iota, 1<<iota - 1  // bit0 == 1, mask0 == 0  (iota == 0)
    bit1, mask1                           // bit1 == 2, mask1 == 1  (iota == 1)
    _, _                                  //                        (iota == 2, unused)
    bit3, mask3                           // bit3 == 8, mask3 == 7  (iota == 3)
)

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

var (
    f1 func()
    f2 func(x int) int
    f3 func(a, _ int, z float32) bool
    f4 func(a, b int, z float32) (bool)
    f5 func(prefix string, values ...int)
    f6 func(a, b int, z float64, opt ...interface{}) (success bool)
    f7 func(int, int, float64) (float64, *[]int)
    f8 func(n int) func(p *T)
)

var mapLiteral = map[string]map[string]string {
    "asd": {
        "qwe": "zxc",
    },
}

func IndexRune(s string, r rune) int {
    for i, c := range s {
        if c == r {
            return i
        }
    }
    for a < b {
        a *= 2
    }
    for i := 0; i < 10; i++ {
        f(i)
    }
    return -1
}

func min(x int, y int) int {
    if x < y {
        return x
    }
    f(a, b, c...);
    if x := f(); x < y {
        return x
    } else if x > z {
        return z
    } else {
        return y
    }
    return y
}

func flushICache(begin, end uintptr)  // implemented externally

func foo() {
    select {
    case i1 = <-c1:
        print("received ", i1, " from c1\n")
    case c2 <- i2:
        print("sent ", i2, " to c2\n")
    case i3, ok := (<-c3):  // same as: i3, ok := <-c3
        if ok {
            print("received ", i3, " from c3\n")
        } else {
            print("c3 is closed\n")
        }
    case a[f()] = <-c4:
        // same as:
        // case t := <-c4
        //    a[f()] = t
    default:
        print("no communication\n")
    }
    
    for {  // send random sequence of bits to c
        select {
        case c <- 0:  // note: no statement, no fallthrough, no folding of cases
        case c <- 1:
        }
    }
    
    select {}  // block forever
    
    switch i := x.(type) {
    case nil:
        printString("x is nil")                // type of i is type of x (interface{})
    case int:
        printInt(i)                            // type of i is int
    case float64:
        printFloat64(i)                        // type of i is float64
    case func(int) float64:
        printFunction(i)                       // type of i is func(int) float64
    case bool, string:
        printString("type is bool or string")  // type of i is type of x (interface{})
    default:
        printString("don't know the type")     // type of i is type of x (interface{})
    }
    
    switch i := foo(); i.(type) {}

    switch tag {
    default: s3()
    case 0, 1, 2, 3: s1()
    case 4, 5, 6, 7: s2()
    }

    switch x := f(); {  // missing switch expression means "true"
    case x < 0: return -x
    default: return x
    }

    switch {
    case x < y: f1()
    case x < z: f2()
    case x == 4: f3()
    }
    
# blanks are permitted in filenames, here the filename is " a:100 " (excluding quotes)
//line  a:100 :10

# colons are permitted in filenames, here the filename is C:foo.go, and the line is 10      
//line C:foo.go:10

# the filename is foo.go, and the line number is 10 for the next line
//line foo.go:10

# the position of x is in the current file with line number 10 and column number 20 
_ = /*line :10:20*/x

# this comment is recognized as invalid line directive (extra blanks around line number)
/*line foo: 10 */

}

//go:nosplit
//go:noescape
func foobar()

//go:linkname baz importpath.name
func baz() {}
"""

class TestParser(unittest.TestCase):
    def test_import(self):
        Parser(Tokenizer(_import_src, 'test.go')).parse()

    def test_const(self):
        print(Parser(Tokenizer(_const_src, 'test.go')).parse())

    def test_line_number(self):
        src = r"""package parser

import (
    `io`
    `net/url`
    `reflect`

    `code.example.org/chenzhuoyu/infra-kernels/utils`

    _ `git.example.org/ee/people/infra/gateway/biz/dispatch/service`
)
"""
        print(Parser(Tokenizer(src, 'test.go')).parse())

if __name__ == '__main__':
    unittest.main()
