# -*- coding: utf-8 -*-

from typing import List
from typing import Tuple
from typing import Union
from typing import TypeVar
from typing import Callable
from typing import Optional

from .ast import VarSpec
from .ast import TypeSpec
from .ast import Function
from .ast import ConstSpec
from .ast import ImportSpec

from .ast import Name
from .ast import String
from .ast import Package
from .ast import Expression

from .ast import Type
from .ast import MapType
from .ast import ArrayType
from .ast import SliceType
from .ast import NamedType
from .ast import StructType
from .ast import ChannelType
from .ast import PointerType
from .ast import FunctionType
from .ast import InterfaceType

from .ast import StructField
from .ast import ChannelDirection

from .tokenizer import State
from .tokenizer import Token
from .tokenizer import Tokenizer

from .tokenizer import TokenType
from .tokenizer import TokenValue

END_TOKENS = {
    TokenType.Int,
    TokenType.Rune,
    TokenType.Name,
    TokenType.Float,
    TokenType.String,
    TokenType.Complex,
}

END_KEYWORDS = {
    'break',
    'return',
    'continue',
    'fallthrough',
}

END_OPERATORS = {
    ')',
    ']',
    '}',
    '++',
    '--',
}

TYPE_KEYWORDS = {
    'map',
    'chan',
    'func',
    'struct',
    'interface',
}

TYPE_OPERATORS = {
    '*',
    '[',
    '<-',
}

def _is_end_token(tk: Token):
    return (tk.kind in END_TOKENS) or \
           (tk.kind == TokenType.Keyword and tk.value in END_KEYWORDS) or \
           (tk.kind == TokenType.Operator and tk.value in END_OPERATORS)

class Parser:
    lx   : Tokenizer
    iota : int
    last : Optional[Token]
    prev : Optional[Token]

    def __init__(self, lx: Tokenizer):
        self.lx = lx
        self.iota = 0
        self.last = None
        self.prev = None

    ### Tokenizer Interfaces ###

    def _read(self) -> Token:
        token = self.lx.next()
        state = self.lx.save_state()

        # not a new line
        if token.kind != TokenType.LF:
            self.last = token
            return token

        # automatic colon insertion rule
        # refs: https://golang.org/ref/spec#Semicolons
        if self.last and _is_end_token(self.last):
            self.last = None
            return Token(state, self.lx.file, TokenType.Operator, ';')

        # otherwise, just skip the new line
        self.last = self.lx.next(ignore_nl = True)
        return self.last

    def _next(self) -> Token:
        if self.prev is None:
            return self._read()
        else:
            ret, self.prev = self.prev, None
            return ret

    def _peek(self) -> Token:
        if self.prev is not None:
            return self.prev
        else:
            self.prev = self._read()
            return self.prev

    def _error(self, tk: Token, msg: str) -> SyntaxError:
        return SyntaxError('%s:%d:%d: %s' % (tk.file, tk.row + 1, tk.col + 1, msg))

    def _should(self, tk: Token, kind: TokenType, value: TokenValue = None) -> bool:
        return tk.kind == kind and (value is None or tk.value == value)

    def _require(self, tk: Token, kind: TokenType, value: TokenValue = None) -> Token:
        if tk.kind != kind:
            raise self._error(tk, '%s expected' % repr(kind.name.lower()))
        elif value is not None and tk.value != value:
            raise self._error(tk, '%s %r expected' % (kind.name.lower(), str(value)))
        else:
            return tk

    def _semicolon(self):
        tk = self._peek()
        tt, tv = tk.kind, tk.value

        # must be either ')', '}' or ';'
        if tt != TokenType.Operator or tv not in (';', ')', '}'):
            raise self._error(tk, '\';\', \')\', \'}\' or new line expected')

        # skip the actual ';'
        if tv == ';':
            self._next()

    ### State Management ###

    def save_state(self) -> Tuple[State, Token, Token, int]:
        return self.lx.save_state(), self.last, self.prev, self.iota

    def load_state(self, state: Tuple[State, Token, Token, int]):
        st, self.last, self.prev, self.iota = state
        self.lx.load_state(st)

    ### Helper Functions ###

    def _is_type(self) -> bool:
        tk = self._peek()
        tt, tv = tk.kind, tk.value

        # check for possible types
        if tt == TokenType.Name:
            return True
        elif tt == TokenType.Keyword:
            return tv in TYPE_KEYWORDS
        elif tt == TokenType.Operator:
            return tv in TYPE_OPERATORS
        else:
            return False

    def _add_names(self, ret: List[StructField], vt: Type, names: List[Token]):
        for name in names:
            self._add_field(ret, vt, name, name = name)

    def _add_field(self, ret: List[StructField], vt: Type, tk: Token, name: Optional[Token] = None):
        sf = StructField(tk)
        sf.type = vt

        # set the field name, if any
        if name is not None:
            sf.name = Name(name)
            sf.name.name = name.value

        # resolve all names
        fname = self._resolve_field_name(sf)
        names = (self._resolve_field_name(fp) for fp in ret)

        # check for duplications
        if fname == '_' or all((fname != item for item in names)):
            ret.append(sf)
        else:
            raise self._error(tk, 'duplicated field %s' % repr(fname))

    def _make_named_ptr(self, tk: Token, vt: NamedType) -> PointerType:
        ret = PointerType(tk)
        ret.base = vt
        return ret

    def _resolve_field_name(self, sf: StructField) -> str:
        if sf.name is not None:
            return sf.name.name
        elif isinstance(sf.type, NamedType):
            return sf.type.name.name
        else:
            assert isinstance(sf.type, PointerType)
            assert isinstance(sf.type.base, NamedType)
            return sf.type.base.name.name

    ### Atomic Parsers ###

    def _parse_name(self) -> Name:
        tk = self._next()
        ret = Name(self._require(tk, TokenType.Name))
        ret.name = tk.value
        return ret

    ### Language Structures --- Types ###

    def _parse_type(self) -> Type:
        if self._should(self._peek(), TokenType.Keyword, 'map'):
            return self._parse_map_type()
        elif self._should(self._peek(), TokenType.Operator, '['):
            return self._parse_list_type()
        elif self._should(self._peek(), TokenType.Name):
            return self._parse_named_type()
        elif self._should(self._peek(), TokenType.Keyword, 'struct'):
            return self._parse_struct_type()
        elif self._should(self._peek(), TokenType.Keyword, 'chan'):
            return self._parse_channel_type(chan = True)
        elif self._should(self._peek(), TokenType.Operator, '<-'):
            return self._parse_channel_type(chan = False)
        elif self._should(self._peek(), TokenType.Operator, '*'):
            return self._parse_pointer_type()
        elif self._should(self._peek(), TokenType.Keyword, 'func'):
            return self._parse_function_type()
        elif self._should(self._peek(), TokenType.Keyword, 'interface'):
            return self._parse_interface_type()
        else:
            raise self._error(self._peek(), 'type specifier expected')

    def _parse_map_type(self) -> MapType:
        ret = MapType(self._next())
        ret.key = self._parse_type()
        self._require(self._next(), TokenType.Operator, ']')
        ret.elem = self._parse_type()
        return ret

    def _parse_list_type(self) -> Union[ArrayType, SliceType]:
        tk = self._next()
        ret = SliceType(tk)

        # check for array length specifier
        if self._should(self._peek(), TokenType.Operator, ']'):
            self._next()
        else:
            ret = ArrayType(tk)
            ret.len = self._parse_expression()
            self._require(self._next(), TokenType.Operator, ']')

        # parse element type
        ret.elem = self._parse_type()
        return ret

    def _parse_named_type(self) -> NamedType:
        tk = self._next()
        ret = NamedType(tk)

        # assume it's a bare name
        ret.name = Name(tk)
        ret.name.name = tk.value

        # check for qualified identifer
        if not self._should(self._peek(), TokenType.Operator, '.'):
            return ret

        # it is a full qualified type name
        self._next()
        ret.package, ret.name = ret.name, self._parse_name()
        return ret

    def _parse_field_decl(self, ret: List[StructField]):
        p = len(ret)
        tk = self._peek()
        state = self.save_state()

        # embedded pointer type
        if self._should(tk, TokenType.Operator, '*'):
            self._next()
            self._require(self._peek(), TokenType.Name)
            self._add_field(ret, self._make_named_ptr(tk, self._parse_named_type()), tk)

        # try parsing as identifier list
        elif self._should(tk, TokenType.Name):
            tk = self._next()
            names = [self._require(tk, TokenType.Name)]

            # parse each identifier
            while self._should(self._peek(), TokenType.Operator, ','):
                self._next()
                names.append(self._require(self._next(), TokenType.Name))

            # normal fields
            if self._is_type():
                self._add_names(ret, self._parse_type(), names)

            # not followed by a type, probably an embedded field
            # restore the state, and re-parse as a named type
            else:
                self.load_state(state)
                self._require(self._peek(), TokenType.Name)
                self._add_field(ret, self._parse_named_type(), tk)

        # otherwise it's an error
        else:
            raise self._error(tk, 'identifier or type specifier expected')

        # check for optional tags
        if self._should(self._peek(), TokenType.String):
            tk = self._next()
            tags = String(tk)

            # update to all fields
            for field in ret[p:]:
                field.tags = tags

    def _parse_struct_type(self) -> StructType:
        ret = StructType(self._next())
        self._require(self._next(), TokenType.Operator, '{')

        # parse every field
        while not self._should(self._peek(), TokenType.Operator, '}'):
            self._parse_field_decl(ret.fields)
            self._semicolon()

        # skip the '}' operator
        self._next()
        return ret

    def _parse_channel_type(self, chan: bool) -> ChannelType:
        tk = self._next()
        ret = ChannelType(tk)

        # <- chan <type>
        if not chan:
            ret.dir = ChannelDirection.Recv
            self._require(self._next(), TokenType.Keyword, 'chan')

        # chan <- <type>
        elif self._should(self._peek(), TokenType.Operator, '<-'):
            self._next()
            ret.dir = ChannelDirection.Send

        # chan <type>
        ret.elem = self._parse_type()
        return ret

    def _parse_pointer_type(self) -> PointerType:
        ret = PointerType(self._next())
        ret.base = self._parse_type()
        return ret

    def _parse_function_type(self) -> FunctionType:
        pass    # TODO: function type

    def _parse_interface_type(self) -> InterfaceType:
        pass    # TODO: interface type

    ### Language Structures --- Expressions ###

    def _parse_expression(self) -> Expression:
        pass    # TODO: expression

    ### Top Level Parsers --- Functions & Methods ###

    def _parse_function(self, ret: List[Function]):
        pass    # TODO: function

    ### Top Level Parsers --- Variables, Types, Constants & Imports ###

    def _parse_var_spec(self, tk: Token, ret: List[VarSpec]):
        pass    # TODO: var spec

    def _parse_type_spec(self, tk: Token, ret: List[TypeSpec]):
        ts = TypeSpec(tk)
        ts.name = Name(self._require(tk, TokenType.Name))
        ts.name.name = tk.value

        # check for duplication
        if ts.name.name != '_':
            if any((ts.name.name == item.name for item in ret)):
                raise self._error(tk, 'duplicated type %s' % repr(ts.name.name))

        # check for type aliasing
        if self._should(self._peek(), TokenType.Operator, '='):
            self._next()
            ts.is_alias = True

        # parse the actual type
        ts.type = self._parse_type()
        ret.append(ts)

    def _parse_const_spec(self, tk: Token, ret: List[ConstSpec]):
        vals = []
        ctype = None
        names = [self._require(tk, TokenType.Name)]

        # const a, b, c, ...
        while self._should(self._peek(), TokenType.Operator, ','):
            self._next()
            names.append(self._require(self._next(), TokenType.Name))

        # optional const type
        if self._is_type():
            ctype = self._parse_type()

        # the "=" operator
        self._require(self._next(), TokenType.Operator, '=')
        vals.append(self._parse_expression())

        # ... = 1, 2, 3, ...
        while self._should(self._peek(), TokenType.Operator, ','):
            self._next()
            vals.append(self._parse_expression())

        # count must match
        if len(vals) != len(names):
            raise self._error(self._peek(), 'cannot assign %d value%s to %d identifier%s' % (
                len(vals)  , '' if len(vals) == 1 else 's',
                len(names) , '' if len(names) == 1 else 's',
            ))

        # add to const table
        for name, value in zip(names, vals):
            if name.value != '_':
                if any((name.value == item.value for item in ret)):
                    raise self._error(name, 'duplicated const %s' % repr(name.value))

            # create a new const spec
            val = ConstSpec(name)
            val.name = Name(name)
            val.name.name = name.value

            # set spec type and value
            val.type = ctype
            val.value = value
            ret.append(val)

    def _parse_import_spec(self, tk: Token, ret: List[ImportSpec]):
        imp = ImportSpec(tk)
        tt, tv = tk.kind, tk.value

        # import . "xxx" / import xxx "xxx"
        if tt == TokenType.Name or (tv == '.' and tt == TokenType.Operator):
            imp.alias, tk = Name(tk), self._next()
            imp.alias.name = tv

        # add to import list
        imp.path = String(self._require(tk, TokenType.String))
        ret.append(imp)

    ### Groupped Parsers ###

    Result = TypeVar('Result')

    def _iota_reset(self):
        self.iota = 0

    def _iota_increment(self):
        self.iota += 1

    def _parse_declarations(self, ret: Result, parser_func: Callable[[Token, Result], None]):
        if not self._should(self._peek(), TokenType.Operator, '('):
            self._iota_reset()
            parser_func(self._next(), ret)
        else:
            self._next()
            self._iota_reset()
            self._parse_declaration_group(ret, parser_func)
            self._require(self._next(), TokenType.Operator, ')')

    def _parse_declaration_group(self, ret: Result, parser_func: Callable[[Token, Result], None]):
        while not self._should(self._peek(), TokenType.Operator, ')'):
            parser_func(self._next(), ret)
            self._iota_increment()
            self._semicolon()

    ### Generic Parsers ###

    def _parse_decl(self, tk: Token, ret: Package):
        if tk.value == 'func':
            self._parse_function(ret.funcs)
        elif tk.value == 'var':
            self._parse_declarations(ret.vars, self._parse_var_spec)
        elif tk.value == 'type':
            self._parse_declarations(ret.types, self._parse_type_spec)
        elif tk.value == 'const':
            self._parse_declarations(ret.consts, self._parse_const_spec)
        else:
            raise self._error(tk, 'unexpected keyword %s' % repr(tk.value))

    def _parse_package(self, ret: Package):
        ret.name = self._parse_name()

    def parse(self) -> Package:
        tk = self._next()
        tk = self._require(tk, TokenType.Keyword, 'package')

        # parse the package name
        ret = Package(tk)
        self._parse_package(ret)
        self._semicolon()

        # imports go before other declarations
        while self._should(self._peek(), TokenType.Keyword, 'import'):
            self._next()
            self._parse_declarations(ret.imports, self._parse_import_spec)
            self._semicolon()

        # parse other top-level declarations
        while self._should(self._peek(), TokenType.Keyword):
            self._parse_decl(self._next(), ret)
            self._semicolon()

        # must be EOF
        self._require(self._next(), TokenType.End)
        return ret
