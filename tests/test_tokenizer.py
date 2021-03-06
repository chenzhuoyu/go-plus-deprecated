#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest

from goplus.tokenizer import Token
from goplus.tokenizer import Tokenizer
from goplus.tokenizer import TokenType
from goplus.tokenizer import TokenValue

names = """
a
_x9
ThisVariableIsExported
αβ
"""

keywords = """
break        default      func         interface    select
case         defer        go           map          struct
chan         else         goto         package      switch
const        fallthrough  if           range        type
continue     for          import       return       var
"""

operators = """
+    &     +=    &=     &&    ==    !=    (    )
-    |     -=    |=     ||    <     <=    [    ]
*    ^     *=    ^=     <-    >     >=    {    }
/    <<    /=    <<=    ++    =     :=    ,    ;
%    >>    %=    >>=    --    !     ...   .    :
     &^          &^=
"""

integers = """
42
0600
0xBadFace
170141183460469231731687303715884105727
"""

integer_vals = (
    42,
    0o600,
    0xBadFace,
    170141183460469231731687303715884105727,
)

floats = """
0.
.25
72.40
2.71828
072.40  // == 72.40
1.e+0
6.67428e-11
1E6
.12345E+5
"""

float_vals = (
    0.,
    .25,
    72.40,
    2.71828,
    072.40,  # == 72.40
    1.e+0,
    6.67428e-11,
    1E6,
    .12345E+5,
)

complexs = """
0i
011i  // == 11i
0755i  // == 755i
0789i  // == 789i
0.i
2.71828i
1.e+0i
6.67428e-11i
1E6i
.25i
.12345E+5i
"""

complex_vals = (
    0j,
    011j,  # == 11j
    0755j,  # == 755j
    0789j,  # == 789j
    0.j,
    2.71828j,
    1.e+0j,
    6.67428e-11j,
    1E6j,
    .25j,
    .12345E+5j,
)

runes = r"""
'a'
'ä'
'本'
'\t'
'\000'
'\007'
'\377'
'\x07'
'\xff'
'\u12e4'
'\U00101234'
'\''         // rune literal containing single quote character
"""

rune_vals = (
    b'a',
    'ä'.encode('utf-8'),
    '本'.encode('utf-8'),
    b'\t',
    b'\000',
    b'\007',
    b'\377',
    b'\x07',
    b'\xff',
    '\u12e4'.encode('utf-8'),
    '\U00101234'.encode('utf-8'),
    b'\'',        # rune literal containing single quote character
)

rune_errs = r"""
'aa'         // illegal: too many characters
'\xa'        // illegal: too few hexadecimal digits
'\0'         // illegal: too few octal digits
'\uDFFF'     // illegal: surrogate half
'\U00110000' // illegal: invalid Unicode code point
"""

rune_err_vals = (
    'too many characters',
    'too few hexadecimal digits',
    'too few octal digits',
    'surrogate half',
    'invalid Unicode code point',
)

strings = r"""
`abc`                // same as "abc"
"\n"
"\""                 // same as `"`
"Hello, world!\n"
"日本語"
"\u65e5本\U00008a9e"
"\xff\u00FF"
`日本語`                                 // UTF-8 input text as a raw literal
"\u65e5\u672c\u8a9e"                    // the explicit Unicode code points
"\U000065e5\U0000672c\U00008a9e"        // the explicit Unicode code points
"\xe6\x97\xa5\xe6\x9c\xac\xe8\xaa\x9e"  // the explicit UTF-8 bytes
"""

string_vals = (
    b'abc',             # same as "abc"
    b'\n',
    b'\"',              # same as `"`
    b'Hello, world!\n',
    '日本語'.encode('utf-8'),
    '\u65e5本\U00008a9e'.encode('utf-8'),
    b'\xff' + '\u00FF'.encode('utf-8'),
    '日本語'.encode('utf-8'),
    '日本語'.encode('utf-8'),
    '日本語'.encode('utf-8'),
    '日本語'.encode('utf-8'),
)

multiline_string = r"""
`\n
\n`                  // same as "\\n\n\\n"
""".strip()

multiline_string_val = b'\\n\n\\n'

string_errs = r"""
"\uD800"             // illegal: surrogate half
"\U00110000"         // illegal: invalid Unicode code point
"""
string_err_vals = (
    'surrogate half',
    'invalid Unicode code point',
)

# noinspection DuplicatedCode
class TestTokenizer(unittest.TestCase):
    def verify(self, src: str, kind: TokenType, value: TokenValue = None, nonl: bool = True):
        tk = Tokenizer(src, '<test>')
        token = tk.next()
        while token.kind == TokenType.Comments or (nonl and token.kind == TokenType.LF):
            if token.kind == TokenType.Comments and not nonl and '\n' in token.value:
                token = Token.eol(tk)
                break
            else:
                token = tk.next()
        self.assertEqual(kind, token.kind, 'wrong token kind: %s' % token)
        self.assertEqual(value, token.value, 'wrong token value: %s' % token)
        token = tk.next()
        while token.kind == TokenType.Comments or token.kind == TokenType.LF:
            token = tk.next()
        self.assertEqual(TokenType.End, token.kind, 'should be EOF: %s' % token)

    def invalid(self, src: str, exc: str):
        self.assertRaisesRegex(SyntaxError, exc, lambda: Tokenizer(src, '<test>').next())

    def test_comments(self):
        self.verify(r'// test comment', TokenType.End)
        self.verify(r'/* test comment */', TokenType.End)
        self.verify("""/* test \n comment */""", TokenType.LF, nonl = False)

    def test_names(self):
        for val in filter(None, map(str.strip, names.splitlines())):
            self.verify(val, TokenType.Name, val)

    def test_keywords(self):
        for val in keywords.split():
            self.verify(val, TokenType.Keyword, val)

    def test_operators(self):
        for val in filter(None, map(str.strip, operators.split())):
            self.verify(val, TokenType.Operator, val)

    def test_integers(self):
        for val, exp in zip(filter(None, map(str.strip, integers.splitlines())), integer_vals):
            self.verify(val, TokenType.Int, exp)

    def test_floats(self):
        for val, exp in zip(filter(None, map(str.strip, floats.splitlines())), float_vals):
            self.verify(val, TokenType.Float, exp)

    def test_complexs(self):
        for val, exp in zip(filter(None, map(str.strip, complexs.splitlines())), complex_vals):
            self.verify(val, TokenType.Complex, exp)

    def test_runes(self):
        for val, exp in zip(filter(None, map(str.strip, runes.splitlines())), rune_vals):
            self.verify(val, TokenType.Rune, exp)

    def test_rune_errs(self):
        for val, err in zip(filter(None, map(str.strip, rune_errs.splitlines())), rune_err_vals):
            self.invalid(val, err)

    def test_strings(self):
        for val, exp in zip(filter(None, map(str.strip, strings.splitlines())), string_vals):
            self.verify(val, TokenType.String, exp)
        self.verify(multiline_string, TokenType.String, multiline_string_val)

    def test_string_errs(self):
        for val, err in zip(filter(None, map(str.strip, string_errs.splitlines())), string_err_vals):
            self.invalid(val, err)

    def test_ellipsis(self):
        tk = Tokenizer('a.....', '<test>')
        token = tk.next()
        self.assertEqual(TokenType.Name, token.kind)
        self.assertEqual('a', token.value)
        token = tk.next()
        self.assertEqual(TokenType.Operator, token.kind)
        self.assertEqual('...', token.value)
        token = tk.next()
        self.assertEqual(TokenType.Operator, token.kind)
        self.assertEqual('.', token.value)
        token = tk.next()
        self.assertEqual(TokenType.Operator, token.kind)
        self.assertEqual('.', token.value)
        token = tk.next()
        self.assertEqual(TokenType.LF, token.kind)
        token = tk.next()
        self.assertEqual(TokenType.End, token.kind)

    def _run_test(self, src, seq):
        tk = Tokenizer(src, '<test>')
        while seq and not tk.is_eof:
            token = tk.next()
            kind, value, row, col, srow, scol = seq[0]
            self.assertEqual(kind, token.kind, repr(seq[0]))
            self.assertEqual(value, token.value, repr(seq[0]))
            self.assertEqual(row, tk.save.row, repr(seq[0]))
            self.assertEqual(col, tk.save.col, repr(seq[0]))
            self.assertEqual(srow, tk.state.row, repr(seq[0]))
            self.assertEqual(scol, tk.state.col, repr(seq[0]))
            self.assertEqual(row, token.row, repr(seq[0]))
            self.assertEqual(col, token.col, repr(seq[0]))
            seq = seq[1:]
        self.assertTrue(not seq, 'not seq')
        self.assertTrue(tk.is_eof, 'tk.is_eof')

    def test_line_number(self):
        src = r"""`hello, world` `hello,
world`
// this is a test
/* this is a
block comment */
"""
        self._run_test(src, [
            (TokenType.String   , b'hello, world'               , 0,  0, 0, 14),
            (TokenType.String   , b'hello,\nworld'              , 0, 15, 1,  6),
            (TokenType.LF       , None                          , 1,  6, 2,  0),
            (TokenType.Comments , ' this is a test'             , 2,  0, 2, 17),
            (TokenType.LF       , None                          , 2, 17, 3,  0),
            (TokenType.Comments , ' this is a\nblock comment '  , 3,  0, 4, 16),
            (TokenType.LF       , None                          , 4, 16, 5,  0),
        ])

    def test_line_number_by_source(self):
        src = r"""package parser

import (
    `io`
    `net/url`
    `reflect`

    `code.example.org/chenzhuoyu/infra-kernels/utils`

    _ `git.example.org/ee/people/infra/gateway/biz/dispatch/service`
)
"""
        self._run_test(src, [
            (TokenType.Keyword  , 'package'                                                         ,  0,  0,  0,  7),
            (TokenType.Name     , 'parser'                                                          ,  0,  8,  0, 14),
            (TokenType.LF       , None                                                              ,  0, 14,  1,  0),
            (TokenType.LF       , None                                                              ,  1,  0,  2,  0),
            (TokenType.Keyword  , 'import'                                                          ,  2,  0,  2,  6),
            (TokenType.Operator , '('                                                               ,  2,  7,  2,  8),
            (TokenType.LF       , None                                                              ,  2,  8,  3,  0),
            (TokenType.String   , b'io'                                                             ,  3,  4,  3,  8),
            (TokenType.LF       , None                                                              ,  3,  8,  4,  0),
            (TokenType.String   , b'net/url'                                                        ,  4,  4,  4, 13),
            (TokenType.LF       , None                                                              ,  4, 13,  5,  0),
            (TokenType.String   , b'reflect'                                                        ,  5,  4,  5, 13),
            (TokenType.LF       , None                                                              ,  5, 13,  6,  0),
            (TokenType.LF       , None                                                              ,  6,  0,  7,  0),
            (TokenType.String   , b'code.example.org/chenzhuoyu/infra-kernels/utils'                ,  7,  4,  7, 53),
            (TokenType.LF       , None                                                              ,  7, 53,  8,  0),
            (TokenType.LF       , None                                                              ,  8,  0,  9,  0),
            (TokenType.Name     , '_'                                                               ,  9,  4,  9,  5),
            (TokenType.String   , b'git.example.org/ee/people/infra/gateway/biz/dispatch/service'   ,  9,  6,  9, 68),
            (TokenType.LF       , None                                                              ,  9, 68, 10,  0),
            (TokenType.Operator , ')'                                                               , 10,  0, 10,  1),
            (TokenType.LF       , None                                                              , 10,  1, 11,  0),
        ])

if __name__ == '__main__':
    unittest.main()
