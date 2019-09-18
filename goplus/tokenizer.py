# -*- coding: utf-8 -*-

import re

from typing import List
from typing import Tuple
from typing import Union
from typing import Optional

from string import digits
from string import hexdigits
from string import octdigits

from enum import IntEnum
from .trie import build_from

class State:
    col: int
    row: int
    pos: int

    __slots__ = (
        'col',
        'row',
        'pos',
    )

    def __init__(self):
        self.col = 0
        self.row = 0
        self.pos = 0

    def copy(self) -> 'State':
        ret = State()
        ret.col = self.col
        ret.row = self.row
        ret.pos = self.pos
        return ret

class Token:
    col   : int
    row   : int
    file  : str
    kind  : 'TokenType'
    value : 'TokenValue'

    __slots__ = (
        'col',
        'row',
        'file',
        'kind',
        'value',
    )

    def __init__(self, col: int, row: int, fname: str, kind: 'TokenType', value: 'TokenValue'):
        self.col = col
        self.row = row
        self.kind = kind
        self.file = fname
        self.value = value

    def __repr__(self) -> str:
        if self.value is None:
            return '#{%d,%d,%s}' % (self.row + 1, self.col + 1, self.kind.name)
        else:
            return '#{%d,%d,%s=%r}' % (self.row + 1, self.col + 1, self.kind.name, self.value)

    def copy(self) -> 'Token':
        return Token(self.col, self.row, self.file, self.kind, self.value)

    @classmethod
    def eol(cls, tk: 'Tokenizer'):
        return cls(tk.save.col, tk.save.row, tk.file, TokenType.LF, None)

    @classmethod
    def end(cls, tk: 'Tokenizer'):
        return cls(tk.save.col, tk.save.row, tk.file, TokenType.End, None)

    @classmethod
    def int(cls, tk: 'Tokenizer', value: int):
        return cls(tk.save.col, tk.save.row, tk.file, TokenType.Int, value)

    @classmethod
    def rune(cls, tk: 'Tokenizer', value: bytes):
        if len(value) == 1:
            return cls(tk.save.col, tk.save.row, tk.file, TokenType.Rune, value[0])
        else:
            return cls(tk.save.col, tk.save.row, tk.file, TokenType.Rune, ord(value.decode('utf-8')))

    @classmethod
    def ident(cls, tk: 'Tokenizer', value: str):
        if value not in KEYWORDS:
            return cls(tk.save.col, tk.save.row, tk.file, TokenType.Name, value)
        else:
            return cls(tk.save.col, tk.save.row, tk.file, TokenType.Keyword, value)

    @classmethod
    def float(cls, tk: 'Tokenizer', value: float):
        return cls(tk.save.col, tk.save.row, tk.file, TokenType.Float, value)

    @classmethod
    def string(cls, tk: 'Tokenizer', value: bytes):
        return cls(tk.save.col, tk.save.row, tk.file, TokenType.String, value)

    @classmethod
    def complex(cls, tk: 'Tokenizer', value: complex):
        return cls(tk.save.col, tk.save.row, tk.file, TokenType.Complex, value)

    @classmethod
    def operator(cls, tk: 'Tokenizer', value: str):
        return cls(tk.save.col, tk.save.row, tk.file, TokenType.Operator, value)

    @classmethod
    def comments(cls, tk: 'Tokenizer', value: str):
        return cls(tk.save.col, tk.save.row, tk.file, TokenType.Comments, value)

    @classmethod
    def directive(cls, tk: 'Tokenizer', value: 'Directive'):
        return cls(tk.save.col, tk.save.row, tk.file, TokenType.Directive, value)

class NoSplitDirective:
    def __repr__(self) -> str:
        return '#{go:nosplit}'

class NoEscapeDirective:
    def __repr__(self) -> str:
        return '#{go:noescape}'

class LinkNameDirective:
    name: str
    link: str

    def __repr__(self) -> str:
        return '#{go:linkname %s %s}' % (self.name, self.link)

Directive = Union[
    NoSplitDirective,
    NoEscapeDirective,
    LinkNameDirective,
]

class TokenType(IntEnum):
    LF        = 0
    End       = 1
    Int       = 2
    Name      = 3
    Rune      = 4
    Float     = 5
    String    = 6
    Complex   = 7
    Keyword   = 8
    Operator  = 9
    Comments  = 254
    Directive = 255

    # virtual token type, used by type inferrer,
    # thus the tokenizer never yields this token
    Bool      = 65535

TokenValue = Optional[Union[
    int,
    str,
    bool,
    bytes,
    float,
    complex,
    Directive,
]]

KEYWORDS = {
    'break',
    'case',
    'chan',
    'const',
    'continue',
    'default',
    'defer',
    'else',
    'fallthrough',
    'for',
    'func',
    'go',
    'goto',
    'if',
    'import',
    'interface',
    'map',
    'package',
    'range',
    'return',
    'select',
    'struct',
    'switch',
    'type',
    'var',
}

OPERATORS = build_from({
    '+',
    '-',
    '*',
    '/',
    '%',
    '&',
    '|',
    '^',
    '<<',
    '>>',
    '&^',
    '+=',
    '-=',
    '*=',
    '/=',
    '%=',
    '&=',
    '|=',
    '^=',
    '<<=',
    '>>=',
    '&^=',
    '&&',
    '||',
    '<-',
    '++',
    '--',
    '==',
    '<',
    '>',
    '=',
    '!',
    '!=',
    '<=',
    '>=',
    ':=',
    '...',
    '(',
    ')',
    '[',
    ']',
    '{',
    '}',
    ',',
    '.',
    ';',
    ':',
})

STD_ESCAPE = {
    '"'  : b'"',
    "'"  : b"'",
    'a'  : b'\a',
    'b'  : b'\b',
    'f'  : b'\f',
    'n'  : b'\n',
    'r'  : b'\r',
    't'  : b'\t',
    'v'  : b'\v',
    '\\' : b'\\',
}

IDENT_REMS  = re.compile(r'\w', re.U)
IDENT_FIRST = re.compile(r'[^\W\d]', re.U)

class _GotComments(BaseException):
    text: str

    def __init__(self, text: str):
        self.text = text

class _GotDirective(BaseException):
    pragma: Directive

    def __init__(self, pragma: Directive):
        self.pragma = pragma

class Tokenizer:
    src   : str
    file  : str
    save  : State
    state : State

    __slots__ = (
        'src',
        'file',
        'save',
        'state',
    )

    def __init__(self, src: str, fname: str):
        self.src = src
        self.file = fname
        self.save = State()
        self.state = State()

        # force a new-line after source
        if not self.src.endswith('\n'):
            self.src += '\n'

    def _error(self, msg: str) -> SyntaxError:
        return SyntaxError('%s:%d:%d: %s' % (self.file, self.save.row + 1, self.save.col + 1, msg))

    def _curr_char(self) -> str:
        return self.src[self.state.pos]

    def _peek_char(self) -> str:
        if self.is_eof:
            return ''
        else:
            return self._curr_char()

    def _next_char(self) -> str:
        if self.is_eof:
            return ''

        # read current char
        st = self.state
        ch = self.src[st.pos]

        # check for new line
        if ch != '\n':
            st.col += 1
        else:
            st.col = 0
            st.row += 1

        # advance read pointer
        st.pos += 1
        return ch

    def _skip_eol(self) -> str:
        pos = self.state.pos
        end = self.src.find('\n', pos)

        # adjust state
        self.state.pos = end
        self.state.col += end - pos

        # slice directly from source
        # this is way faster than read char by char
        return self.src[pos:end]

    def _skip_char(self) -> str:
        self.save.col = self.state.col
        self.save.row = self.state.row
        self.save.pos = self.state.pos
        return self._next_char()

    def _skip_space(self, ch: str) -> str:
        while ch.isspace() and (ch != '\n'):
            ch = self._skip_char()
        else:
            return ch

    def _skip_blanks(self) -> str:
        while True:
            ch = self._skip_char()
            ch = self._skip_space(ch)

            # unix style comments
            if ch == '#':
                self._skip_eol()
                continue

            # check for possible comments
            if ch != '/' or (self._peek_char() not in ('*', '/')):
                return ch

            # line comments
            if self._next_char() == '/':
                if not self._check_sol() or not self._check_prefix('go:', 'line '):
                    raise _GotComments(self._skip_eol())
                else:
                    self._handle_directives(self._skip_eol(), block = False)
                    continue

            # locate the comment end
            pos = self.state.pos
            end = self.src.find('*/', pos)
            ret = self._read_fast(pos, end, 2)

            # special case for 'line' directive
            if not ret.startswith('line '):
                raise _GotComments(ret)
            else:
                self._handle_directives(ret, block = True)

    def _read_rune(self, ch: str) -> bytes:
        if ch != '\\':
            return ch.encode('utf-8')
        else:
            return self._read_escape(self._next_char())

    def _read_fast(self, pos: int, end: int, tail: int) -> str:
        ret = self.src[pos:end]
        nnl, end = ret.count('\n'), end + tail

        # adjust column based on new-line count
        if nnl > 0:
            self.state.col = -ret.rfind('\n') - 1

        # adjust tokenizer state
        self.state.pos = end
        self.state.row += nnl
        self.state.col += end - pos
        return ret

    def _read_until(self, quote: str, delim: Optional[str] = None) -> Tuple[str, bytes]:
        pos = self.state.pos
        end = self.src.find(quote, pos)

        # also consider delimiter, if any
        if delim is not None:
            dps = self.src.find(delim, pos)
            end = dps if end < 0 else end if dps < 0 else min(dps, end)

        # doesn't find anything
        if end < 0:
            raise self._error('unexpected EOF')

        # slice directly from source
        # this increases performance drastically
        return (
            self.src[end],
            self._read_fast(pos, end, 1).encode('utf-8'),
        )

    def _read_digit(self, name: str, charset: str) -> str:
        if self.is_eof or self._curr_char() not in charset:
            raise self._error('too few %s digits' % name)
        else:
            return self._next_char()

    def _read_digits(self, name: str, charset: str, n: int) -> str:
        return ''.join((self._read_digit(name, charset) for _ in range(n)))

    def _read_escape(self, ch: str) -> bytes:
        if not ch:
            raise self._error('unexpected EOF')
        elif ch == 'u':
            return self._read_escape_uni(4)
        elif ch == 'U':
            return self._read_escape_uni(8)
        elif ch == 'x':
            return self._read_escape_hex()
        elif ch in octdigits:
            return self._read_escape_oct(ch)
        elif ch in STD_ESCAPE:
            return STD_ESCAPE[ch]
        else:
            raise self._error('invalid escape character %s' % repr(ch))

    def _read_escape_hex(self) -> bytes:
        return bytes([int(self._read_digits('hexadecimal', hexdigits, 2), 16)])

    def _read_escape_oct(self, ch: str) -> bytes:
        rem = self._read_digits('octal', octdigits, 2)
        val = int(ch + rem, 8)

        # check for octal range
        if val > 256:
            raise self._error('octal escape value > 255: %d' % val)
        else:
            return bytes([val])

    def _read_escape_uni(self, size: int) -> bytes:
        try:
            return chr(int(self._read_digits('hexadecimal', hexdigits, size), 16)).encode('utf-8')
        except ValueError as e:
            exc = e

        # do not throw the exception in the `except` section
        # we don't want this to be chained to the original `UnicodeError`
        if isinstance(exc, UnicodeError):
            raise self._error('surrogate half')
        else:
            raise self._error('invalid Unicode code point')

    def _check_sol(self) -> bool:
        if self.state.pos == 0:
            return True
        elif self.state.pos < 3:
            return False
        else:
            return self.src[self.state.pos - 3] == '\n'

    def _check_prefix(self, *pfx: str) -> bool:
        for p in pfx:
            if self.src[self.state.pos:self.state.pos + len(p)] == p:
                return True
        else:
            return False

    def _handle_directives(self, cdir: str, block: bool):
        if cdir == 'go:nosplit':
            raise _GotDirective(NoSplitDirective())
        elif cdir == 'go:noescape':
            raise _GotDirective(NoEscapeDirective())
        elif cdir.startswith('line '):
            self._handle_directives_line(cdir, cdir[5:].rsplit(':', 2), block)
        elif cdir.startswith('go:linkname '):
            self._handle_directives_linkname(cdir, list(filter(None, cdir[12:].split(' '))))

    def _handle_directives_line(self, cdir: str, args: List[str], block: bool):
        try:
            row = int(args[-1])
        except ValueError:
            raise _GotComments(cdir)

        # check tokenizer state
        if not block and self._next_char() != '\n':
            raise SystemError('incorrect tokenizer state')

        # row number must be greater than 0
        if row <= 0:
            raise _GotComments(cdir)

        # adjust the row number
        st = self.state
        st.row = row - 1

        # check for column number
        if len(args) <= 2 or not args[-2].isdigit() or int(args[-2]) <= 0:
            name = ':'.join(args[:-1])
        else:
            name = ':'.join(args[:-2])
            st.row, st.col = int(args[-2]) - 1, st.row

        # check for file name
        if name:
            self.file = name

    def _handle_directives_linkname(self, cdir: str, args: List[str]):
        if len(args) != 2:
            raise _GotComments(cdir)
        else:
            ret = LinkNameDirective()
            ret.name, ret.link = args
            raise _GotDirective(ret)

    def _parse(self, ch: str) -> Token:
        if not ch:
            return Token.end(self)
        elif ch == '\n':
            return Token.eol(self)
        elif ch == '`':
            return self._parse_raw()
        elif ch == "'":
            return self._parse_rune()
        elif ch == '"':
            return self._parse_string()
        elif ch == '.':
            return self._parse_decimal()
        elif ch in digits:
            return self._parse_number(ch)
        elif ch in OPERATORS:
            return self._parse_operator(ch)
        elif IDENT_FIRST.match(ch):
            return self._parse_identifier(ch)
        else:
            raise self._error('invalid character %s' % repr(ch))

    def _parse_raw(self) -> Token:
        _, val = self._read_until('`')
        return Token.string(self, val)

    def _parse_rune(self) -> Token:
        val = self._next_char()
        ret = self._read_rune(val)

        # rune must contains exact 1 character
        if val == "'":
            raise self._error('empty rune')
        elif self._next_char() != "'":
            raise self._error('too many characters')
        else:
            return Token.rune(self, ret)

    def _parse_string(self) -> Token:
        ret = b''
        nch, buf = self._read_until('"', '\\')

        # scan until end of quote
        while nch != '"':
            ret += buf
            ret += self._read_escape(self._next_char())
            nch, buf = self._read_until('"', '\\')

        # build the token
        return Token.string(self, ret + buf)

    def _parse_number(self, first: str) -> Token:
        ret = first
        nch = self._peek_char()

        # special case of leading zero
        if first == '0':
            if not nch:
                return Token.int(self, 0)
            elif nch in 'bB':
                self._next_char()
                return self._parse_number_charset('binary', 2, '01')
            elif nch in 'oO':
                self._next_char()
                return self._parse_number_charset('octal', 8, octdigits)
            elif nch in 'xX':
                self._next_char()
                return self._parse_number_charset('hex', 16, hexdigits)

        # scan each digit
        while nch and nch in digits:
            ret += self._next_char()
            nch = self._peek_char()

        # check for decimal point, complex numbers and scientific notation
        if nch == '.':
            return self._parse_number_float(ret)
        elif nch == 'i':
            return self._parse_number_complex(ret)
        elif nch in ('e', 'E'):
            return self._parse_number_science(ret)
        elif ret[0] != '0':
            return Token.int(self, int(ret, 10))
        elif all((x in octdigits for x in ret)):
            return Token.int(self, int(ret, 8))
        else:
            raise self._error('invalid octal digit')

    def _parse_number_float(self, first: str) -> Token:
        rem = self._next_char()
        nch = self._peek_char()

        # scan every digit
        while nch and nch in digits:
            rem += self._next_char()
            nch = self._peek_char()

        # check for complex numbers, and scientific notation
        if nch == 'i':
            return self._parse_number_complex(first + rem)
        elif nch in ('e', 'E'):
            return self._parse_number_science(first + rem)
        else:
            return Token.float(self, float(first + rem))

    def _parse_number_complex(self, first: str) -> Token:
        self._next_char()
        return Token.complex(self, float(first) * 1j)

    def _parse_number_science(self, first: str) -> Token:
        ok = False
        ret = self._next_char()
        nch = self._peek_char()

        # check for exponent sign
        if nch in ('+', '-'):
            ret += self._next_char()
            nch = self._peek_char()

        # scan every digit
        while nch and nch in digits:
            ret += self._next_char()
            nch, ok = self._peek_char(), True

        # check for complex numbers
        if not ok:
            raise self._error('invalid float literal')
        elif nch == 'i':
            return self._parse_number_complex(first + ret)
        else:
            return Token.float(self, float(first + ret))

    def _parse_number_charset(self, name: str, base: int, charset: str) -> Token:
        ret = ''
        nch = self._peek_char()

        # scan every digit
        while nch and nch in charset:
            ret += self._next_char()
            nch = self._peek_char()

        # check for result
        if not ret or (nch and nch in hexdigits):
            raise self._error('invalid %s digit' % name)
        else:
            return Token.int(self, int(ret, base))

    def _parse_decimal(self) -> Token:
        if self.is_eof or self._curr_char() not in digits:
            return self._parse_operator('.')
        else:
            return self._parse_number_float('.')

    def _parse_operator(self, first: str) -> Token:
        ops = first
        node = OPERATORS.children[first]

        # in Golang, the first character of every
        # multi-char operator is a valid operator by it's own
        if not node.is_leaf:
            raise SystemError('invalid operator table')

        # commit the first char
        ret = ops
        pst = self.save_state()
        nch = self._peek_char()

        # traverse the trie tree
        while nch and nch in node:
            node = node.children[nch]
            ops += self._next_char()
            nch = self._peek_char()

            # commit the current state for leaf nodes
            if node.is_leaf:
                ret = ops
                pst = self.save_state()

        # build the token
        self.load_state(pst)
        return Token.operator(self, ret)

    def _parse_identifier(self, first: str) -> Token:
        ret = first
        nch = self._peek_char()

        # scan until no more identifier characters
        while nch and IDENT_REMS.match(nch):
            ret += self._next_char()
            nch = self._peek_char()

        # build the token
        return Token.ident(self, ret)

    @property
    def is_eof(self):
        return self.state.pos >= len(self.src)

    def next(self) -> Token:
        try:
            nch = self._skip_blanks()
        except _GotComments as e:
            return Token.comments(self, e.text)
        except _GotDirective as e:
            return Token.directive(self, e.pragma)
        else:
            return self._parse(nch)

    def save_state(self) -> State:
        return self.state.copy()

    def load_state(self, state: State):
        self.state = state.copy()
