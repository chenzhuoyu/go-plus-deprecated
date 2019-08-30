# -*- coding: utf-8 -*-

from typing import Set
from typing import List
from typing import Type as Tp
from typing import Tuple
from typing import Union
from typing import Callable
from typing import Optional

from .ast import LinkSpec
from .ast import InitSpec
from .ast import TypeSpec
from .ast import Function
from .ast import ImportSpec

from .ast import Int
from .ast import Name
from .ast import Rune
from .ast import Float
from .ast import String
from .ast import Complex
from .ast import Operator

from .ast import Package
from .ast import Primary
from .ast import Conversion
from .ast import Expression

from .ast import Go
from .ast import If
from .ast import For
from .ast import Goto
from .ast import Break
from .ast import Defer
from .ast import Label
from .ast import Return
from .ast import ForRange
from .ast import Continue

from .ast import Switch
from .ast import SwitchCase

from .ast import TypeSwitch
from .ast import TypeSwitchCase

from .ast import Select
from .ast import SelectCase
from .ast import SelectReceive

from .ast import Send
from .ast import Empty
from .ast import IncDec
from .ast import Statement
from .ast import Assignment
from .ast import Fallthrough

from .ast import SimpleStatement
from .ast import CompoundStatement

from .ast import Lambda
from .ast import Element
from .ast import Operand
from .ast import Composite
from .ast import LiteralType
from .ast import LiteralValue
from .ast import VarArrayType

from .ast import Index
from .ast import Slice
from .ast import Selector
from .ast import Arguments
from .ast import Assertion

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

from .ast import ImportHere
from .ast import ChannelOptions

from .ast import StructField
from .ast import InterfaceMethod

from .ast import FunctionOptions
from .ast import FunctionArgument
from .ast import FunctionSignature

from .tokenizer import State
from .tokenizer import Token
from .tokenizer import Tokenizer

from .tokenizer import TokenType
from .tokenizer import TokenValue

from .tokenizer import NoSplitDirective
from .tokenizer import NoEscapeDirective
from .tokenizer import LinkNameDirective

LIT_NODES = {
    TokenType.Int     : Int,
    TokenType.Rune    : Rune,
    TokenType.Float   : Float,
    TokenType.String  : String,
    TokenType.Complex : Complex,
}

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

UNARY_OPERATORS = {
    '+',
    '-',
    '!',
    '^',
    '*',
    '&',
    '<-',
}

BINARY_OPERATORS = [
    {'||'},
    {'&&'},
    {'==', '!=', '<', '<=', '>', '>='},
    {'+' , '-' , '|', '^'},
    {'*' , '/' , '%', '<<', '>>', '&', '&^'},
]

INCDEC_OPERATORS = {
    '++',
    '--',
}

RETURN_OPERATORS = {
    ';',
    ')',
    '}',
}

ARGUMENT_OPERATORS = {
    ',',
    ')',
    '...',
}

ASSIGNMENT_OPERATORS = {
    '=',
    '+=',
    '-=',
    '*=',
    '/=',
    '%=',
    '&=',
    '|=',
    '^=',
    '&^=',
    '<<=',
    '>>=',
}

def _is_end_token(tk: Token):
    return (tk.kind in END_TOKENS) or \
           (tk.kind == TokenType.Keyword and tk.value in END_KEYWORDS) or \
           (tk.kind == TokenType.Operator and tk.value in END_OPERATORS)

PState = Tuple[
    State,
    Token,
    Token,
    int,
    int,
    FunctionOptions,
]

class Parser:
    lx     : Tokenizer
    expr   : int
    iota   : int
    last   : Optional[Token]
    prev   : Optional[Token]
    fflags : FunctionOptions

    class _Scope:
        ps: 'Parser'
        st: Optional[PState]

        def __init__(self, ps: 'Parser'):
            self.ps = ps
            self.st = None

        def __exit__(self, exc_type, exc_val, exc_tb):
            if self.st is not None:
                self.ps.load_state(self.st)
                self.st = None

        def __enter__(self):
            self.st = self.ps.save_state()
            return self

    class _Nested:
        ps: 'Parser'

        def __init__(self, ps: 'Parser'):
            self.ps = ps

        def __exit__(self, exc_type, exc_val, exc_tb):
            self.ps.expr -= 1

        def __enter__(self):
            self.ps.expr += 1
            return self

    class _Control:
        ps: 'Parser'
        ex: Optional[int]

        def __init__(self, ps: 'Parser'):
            self.ps = ps
            self.ex = None

        def __exit__(self, exc_type, exc_val, exc_tb):
            if self.ex is not None:
                self.ps.expr = self.ex
                self.ex = None

        def __enter__(self):
            self.ex = self.ps.expr
            self.ps.expr = -1
            return self

    def __init__(self, lx: Tokenizer):
        self.lx = lx
        self.expr = 0
        self.iota = 0
        self.last = None
        self.prev = None
        self.fflags = FunctionOptions(0)

    ### Tokenizer Interfaces ###

    def _read(self) -> Token:
        token = self.lx.next()
        state = self.lx.save.copy()

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

    def _delimiter(self, delim: str):
        tk = self._peek()
        tt, tv = tk.kind, tk.value

        # must be either ')', '}' or the delimiter
        if tt != TokenType.Operator or tv not in (delim, ')', '}'):
            raise self._error(tk, '%s or new line expected' % repr(delim))

        # skip the actual delimiter
        if tv == delim:
            self._next()

    ### State Management ###

    def save_state(self) -> PState:
        return self.lx.save_state(), self.last, self.prev, self.iota, self.expr, self.fflags

    def load_state(self, state: PState):
        st, self.last, self.prev, self.iota, self.expr, self.fflags = state
        self.lx.load_state(st)

    ### Helper Functions ###

    def _add_names(self, ret: List[StructField], vt: Type, names: List[Token]):
        for name in names:
            self._add_field(ret, vt, name, name = name)

    def _add_field(self, ret: List[StructField], vt: Type, tk: Token, name: Optional[Token] = None):
        sf = StructField(tk)
        sf.type = vt
        sf.name = Name(name) if name else None
        ret.append(sf)

    def _make_named_ptr(self, tk: Token, vt: NamedType) -> PointerType:
        ret = PointerType(tk)
        ret.base = vt
        return ret

    ### Type Guessing Functions ###

    def _is_svd(self) -> bool:
        with self._Scope(self):
            try:
                self._parse_name()
                self._ensure_svd_names()
                self._require(self._next(), TokenType.Operator, ':=')
            except SyntaxError:
                return False
            else:
                return True

    def _is_ops(self, ops: Set[str]) -> bool:
        tk = self._peek()
        return tk.value in ops and tk.kind == TokenType.Operator

    def _is_label(self) -> bool:
        with self._Scope(self):
            try:
                self._parse_name()
                self._require(self._next(), TokenType.Operator, ':')
            except SyntaxError:
                return False
            else:
                return True

    def _is_case_end(self) -> bool:
        if self._should(self._peek(), TokenType.Operator, '}'):
            return True
        elif self._should(self._peek(), TokenType.Keyword, 'case'):
            return True
        elif self._should(self._peek(), TokenType.Keyword, 'default'):
            return True
        else:
            return False

    def _is_type_expr(self) -> bool:
        with self._Scope(self):
            if not self._should(self._next(), TokenType.Operator, '.'):
                return False
            elif not self._should(self._next(), TokenType.Operator, '('):
                return False
            elif not self._should(self._next(), TokenType.Keyword, 'type'):
                return False
            elif not self._should(self._next(), TokenType.Operator, ')'):
                return False
            else:
                return True

    def _is_named_type(self) -> bool:
        with self._Scope(self):
            try:
                self._parse_named_type()
            except SyntaxError:
                return False
            else:
                return True

    def _is_literal_type(self) -> bool:
        with self._Scope(self):
            try:
                self._parse_literal_type()
                self._require(self._next(), TokenType.Operator, '{')
            except SyntaxError:
                return False
            else:
                return True

    def _is_argument_type(self) -> bool:
        with self._Scope(self):
            try:
                self._parse_type()
            except SyntaxError:
                return False
            else:
                return self._is_ops(ARGUMENT_OPERATORS)

    def _is_probably_type(self) -> bool:
        tk = self._peek()
        tt, tv = tk.kind, tk.value

        # check for possible types
        if tt == TokenType.Name:
            return True

        # maybe: map, chan, func, struct, interface
        if tt == TokenType.Keyword:
            return tv in TYPE_KEYWORDS

        # then it must be an operator
        if tt != TokenType.Operator:
            return False

        # special case for '(' operator
        if tv != '(':
            return tv in TYPE_OPERATORS

        # try match rule :: Type = '(' Type ')' .
        with self._Scope(self):
            self._next()
            return self._is_probably_type()

    def _is_conversion_type(self) -> bool:
        with self._Scope(self):
            try:
                self._ensure_not_names()
                self._require(self._next(), TokenType.Operator, '(')
            except SyntaxError:
                return False
            else:
                return True

    def _ensure_svd_names(self):
        while self._should(self._peek(), TokenType.Operator, ','):
            self._next()
            self._parse_name()

    def _ensure_not_names(self):
        if isinstance(self._parse_type(), NamedType):
            raise SyntaxError('named types')

    def _ensure_type_switch(self):
        self._require(self._next(), TokenType.Operator, '.')
        self._require(self._next(), TokenType.Operator, '(')
        self._require(self._next(), TokenType.Keyword, 'type')
        self._require(self._next(), TokenType.Operator, ')')

    def _ensure_receiver_args(self, args: List[FunctionArgument]) -> FunctionArgument:
        if len(args) != 1:
            raise self._error(self._peek(), 'method has multiple receivers')
        else:
            return args[0]

    ### Expression Prunning Functions ###

    def _prune_expression_tree(self, expr: Expression) -> Expression:
        if isinstance(expr.left, Primary):
            expr = self._prune_expression_primary(expr)
        else:
            expr.left = self._prune_expression_tree(expr.left)

        # prune right expression
        if expr.right is not None:
            expr.right = self._prune_expression_tree(expr.right)

        # promote the left term if no operator and right term present
        if expr.op is not None:
            return expr
        elif expr.right is not None:
            return expr
        elif isinstance(expr.left, Primary):
            return expr
        else:
            return expr.left

    def _prune_expression_primary(self, expr: Expression) -> Expression:
        for mod in expr.left.mods:
            if isinstance(mod, Index):
                self._prune_expression_primary_index(mod)
            elif isinstance(mod, Slice):
                self._prune_expression_primary_slice(mod)
            elif isinstance(mod, Arguments):
                self._prune_expression_primary_arguments(mod)
        else:
            if not isinstance(expr.left.val, Expression):
                return expr
            elif not expr.left.mods:
                return self._prune_expression_tree(expr.left.val)
            else:
                expr.left.val = self._prune_expression_tree(expr.left.val)
                return expr

    def _prune_expression_primary_index(self, mod: Index):
        mod.expr = self._prune_expression_tree(mod.expr)

    def _prune_expression_primary_slice(self, mod: Slice):
        mod.pos = mod.pos and self._prune_expression_tree(mod.pos)
        mod.len = mod.len and self._prune_expression_tree(mod.len)
        mod.cap = mod.cap and self._prune_expression_tree(mod.cap)

    def _prune_expression_primary_arguments(self, mod: Arguments):
        for i, arg in enumerate(mod.args):
            if isinstance(arg, Expression):
                mod.args[i] = self._prune_expression_tree(arg)

    ### Basic Structure Parsers ###

    def _parse_svd(self) -> List[Name]:
        val = self._parse_name()
        ret = [val]

        # may have multiple names
        while self._should(self._peek(), TokenType.Operator, ','):
            self._next()
            ret.append(self._parse_name())

        # the it must be ':='
        self._require(self._next(), TokenType.Operator, ':=')
        return ret

    def _parse_name(self) -> Name:
        return Name(self._require(self._next(), TokenType.Name))

    def _parse_case(
        self,
        name   : str,
        klass  : Tp[Union[SelectCase, SwitchCase, TypeSwitchCase]],
        cases  : List[Union[SelectCase, SwitchCase, TypeSwitchCase]],
        parser : Callable[[Union[SelectCase, SwitchCase, TypeSwitchCase]], None],
    ):
        default = False
        self._require(self._next(), TokenType.Operator, '{')

        # parse every case
        while not self._should(self._peek(), TokenType.Operator, '}'):
            default = self._parse_case_item(name, klass, cases, parser, default)
        else:
            self._next()

    def _parse_case_body(self, body: List[Statement]):
        while not self._is_case_end():
            body.append(self._parse_statement())
            self._delimiter(';')

    def _parse_case_item(
        self,
        name    : str,
        klass   : Tp[Union[SelectCase, SwitchCase, TypeSwitchCase]],
        cases   : List[Union[SelectCase, SwitchCase, TypeSwitchCase]],
        parser  : Callable[[Union[SelectCase, SwitchCase, TypeSwitchCase]], None],
        default : bool,
    ) -> bool:
        tk = self._next()
        ok = self._should(tk, TokenType.Keyword, 'case')
        ret = klass(tk)

        # check for 'default' case
        if ok:
            parser(ret)
        elif not self._should(tk, TokenType.Keyword, 'default'):
            raise self._error(tk, '\'case\' or \'default\' expected')
        elif default:
            raise self._error(tk, 'multiple defaults in %s' % name)

        # parse the case body
        self._require(self._next(), TokenType.Operator, ':')
        self._parse_case_body(ret.body)

        # add to cases
        cases.append(ret)
        return not ok

    def _parse_types(self) -> List[Type]:
        val = self._parse_type()
        ret = [val]

        # may have multiple names
        while self._should(self._peek(), TokenType.Operator, ','):
            self._next()
            ret.append(self._parse_type())

        # all done
        return ret

    def _parse_operator(self, ops: Set[str]) -> Operator:
        if not self._is_ops(ops):
            raise self._error(self._peek(), 'invalid operator')
        else:
            return Operator(self._next())

    def _parse_expressions(self) -> List[Expression]:
        val = self._parse_expression()
        ret = [val]

        # may have multiple expressions
        while self._should(self._peek(), TokenType.Operator, ','):
            self._next()
            ret.append(self._parse_expression())

        # all done
        return ret

    def _parse_initializer(self) -> Optional[SimpleStatement]:
        st = self.save_state()
        ret = self._parse_initializer_st()

        # check for the ';' operator
        if ret and self._should(self._next(), TokenType.Operator, ';'):
            return ret

        # no, it's not an init statement
        self.load_state(st)
        return None

    def _parse_initializer_st(self) -> Optional[SimpleStatement]:
        with self._Control(self):
            try:
                return self._parse_simple_statement()
            except SyntaxError:
                return None

    ### Nested Parsers ###

    def _parse_nested_end(self, ret: Union[Type, Expression]) -> Union[Type, Expression]:
        self._require(self._next(), TokenType.Operator, ')')
        return ret

    def _parse_nested_type(self) -> Type:
        self._next()
        return self._parse_nested_end(self._parse_type())

    def _parse_nested_expr(self) -> Expression:
        with self._Nested(self):
            self._next()
            return self._parse_nested_end(self._parse_expression())

    ### Literal Type Parser ###

    def _parse_literal_type(self) -> LiteralType:
        if self._should(self._peek(), TokenType.Keyword, 'map'):
            return self._parse_map_type()
        elif self._should(self._peek(), TokenType.Operator, '['):
            return self._parse_list_type(for_literal = True)
        elif self._should(self._peek(), TokenType.Name):
            return self._parse_named_type()
        elif self._should(self._peek(), TokenType.Keyword, 'struct'):
            return self._parse_struct_type()
        else:
            raise self._error(self._peek(), 'literal type specifier expected')

    ### Language Structures --- Types ###

    def _parse_type(self) -> Type:
        if self._should(self._peek(), TokenType.Keyword, 'map'):
            return self._parse_map_type()
        elif self._should(self._peek(), TokenType.Operator, '['):
            return self._parse_list_type(for_literal = False)
        elif self._should(self._peek(), TokenType.Name):
            return self._parse_named_type()
        elif self._should(self._peek(), TokenType.Operator, '('):
            return self._parse_nested_type()
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
        self._require(self._next(), TokenType.Operator, '[')
        ret.key = self._parse_type()
        self._require(self._next(), TokenType.Operator, ']')
        ret.elem = self._parse_type()
        return ret

    def _parse_list_type(self, for_literal: bool) -> Union[ArrayType, SliceType, VarArrayType]:
        tk = self._next()
        ret = SliceType(tk)

        # slices: [] <type>
        if self._should(self._peek(), TokenType.Operator, ']'):
            self._next()

        # arrays without length: [...] <type>
        elif self._should(self._peek(), TokenType.Operator, '...') and for_literal:
            ret = VarArrayType(tk)
            self._require(self._next(), TokenType.Operator, ']')

        # arrays with length: [<len>] type
        else:
            with self._Nested(self):
                ret = ArrayType(tk)
                ret.len = self._parse_expression()
                self._require(self._next(), TokenType.Operator, ']')

        # parse element type
        ret.elem = self._parse_type()
        return ret

    def _parse_named_type(self) -> NamedType:
        tk = self._next()
        ret = NamedType(tk)
        ret.name = Name(tk)

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

            # normal fields, an approximated guessing is good enough
            if self._is_probably_type():
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
            self._delimiter(';')

        # skip the '}' operator
        self._next()
        return ret

    def _parse_channel_type(self, chan: bool) -> ChannelType:
        ret = ChannelType(self._next())
        ret.dir = ChannelOptions.BOTH

        # <- chan <type>
        if not chan:
            ret.dir = ChannelOptions.RECV
            self._require(self._next(), TokenType.Keyword, 'chan')

        # chan <- <type>
        elif self._should(self._peek(), TokenType.Operator, '<-'):
            self._next()
            ret.dir = ChannelOptions.SEND

        # chan <type>
        ret.elem = self._parse_type()
        return ret

    def _parse_pointer_type(self) -> PointerType:
        ret = PointerType(self._next())
        ret.base = self._parse_type()
        return ret

    def _parse_function_type(self) -> FunctionType:
        ret = FunctionType(self._next())
        ret.type = self._parse_signature()
        return ret

    def _parse_interface_type(self) -> InterfaceType:
        ret = InterfaceType(self._next())
        self._require(self._next(), TokenType.Operator, '{')

        # parse each signature
        while not self._should(self._peek(), TokenType.Operator, '}'):
            ret.decls.append(self._parse_interface_decl())
            self._delimiter(';')

        # skip the '}'
        self._next()
        return ret

    def _parse_interface_decl(self) -> Union[NamedType, InterfaceMethod]:
        st = self.save_state()
        ret = self._parse_named_type()

        # check for named types
        if self._should(self._peek(), TokenType.Operator, ';'):
            return ret

        # no, it's a method declaration
        self.load_state(st)
        return self._parse_interface_method()

    def _parse_interface_method(self) -> InterfaceMethod:
        ret = InterfaceMethod(self._peek())
        ret.name = self._parse_name()
        ret.type = self._parse_signature()
        return ret

    ### Language Structures --- Functions ###

    def _parse_arg_name(self) -> FunctionArgument:
        ret = FunctionArgument(self._peek())
        ret.name = self._parse_name()
        return ret

    def _parse_arg_list(self) -> List[FunctionArgument]:
        arg0 = self._parse_arg_name()
        args = [arg0]

        # parse remaining names
        while self._should(self._peek(), TokenType.Operator, ','):
            self._next()
            args.append(self._parse_arg_name())

        # parse the type
        vt = self._parse_type()
        arg0.type = vt

        # copy the type to every identifier
        for arg in args[1:]:
            arg.type = vt
        else:
            return args

    def _parse_arg_decl(self, tk: Token, for_args: bool) -> Tuple[bool, FunctionArgument]:
        var = False
        arg = FunctionArgument(tk)
        arg.name = self._parse_name()

        # check for variadic parameter
        if self._should(self._peek(), TokenType.Operator, '...'):
            if not for_args:
                raise self._error(self._peek(), 'cannot use ... in receiver or result parameter list')
            else:
                var = True
                self._next()

        # parse the type
        arg.type = self._parse_type()
        return var, arg

    def _parse_signature(self) -> FunctionSignature:
        ret = FunctionSignature(self._peek())
        ret.var, ret.args = self._parse_parameters(for_args = True)

        # multiple return values
        if self._should(self._peek(), TokenType.Operator, '('):
            _, ret.rets = self._parse_parameters(for_args = False)
            return ret

        # no return values
        if not self._is_probably_type():
            return ret

        # single bare-type return value
        rt = FunctionArgument(self._peek())
        rt.type = self._parse_type()
        ret.rets.append(rt)
        return ret

    def _parse_parameters(self, for_args: bool) -> Tuple[bool, List[FunctionArgument]]:
        vn, ret, var = None, [], False
        self._require(self._next(), TokenType.Operator, '(')

        # parse every argument
        while not var and not self._should(self._peek(), TokenType.Operator, ')'):
            tk = self._peek()
            st = self.save_state()

            # try parsing as "name [...]Type"
            if vn is not False:
                try:
                    var, arg = self._parse_arg_decl(tk, for_args)
                except SyntaxError:
                    self.load_state(st)
                else:
                    vn = True
                    ret.append(arg)
                    self._delimiter(',')
                    continue

            # not working, try "name1, name2, ..., nameX Type"
            if vn is not False:
                try:
                    ret.extend(self._parse_arg_list())
                except SyntaxError:
                    self.load_state(st)
                else:
                    vn = True
                    self._delimiter(',')
                    continue

            # not working either, must be a bare type
            # the names must either all be present or all be absent
            if vn is True:
                raise self._error(tk, 'parameter name expected')

            # maybe variadic
            if self._should(tk, TokenType.Operator, '...'):
                if not for_args:
                    raise self._error(tk, 'cannot use ... in receiver or result parameter list')
                else:
                    var = True
                    self._next()

            # parse the type
            vn = False
            arg = FunctionArgument(tk)
            arg.type = self._parse_type()
            ret.append(arg)
            self._delimiter(',')

        # must end with a ')'
        self._require(self._next(), TokenType.Operator, ')')
        return var, ret

    ### Language Structures --- Statements / Basic Structures ###

    def _parse_go(self) -> Go:
        ret = Go(self._next())
        ret.expr = self._parse_expression()

        # must be a function call
        if not ret.expr.is_call():
            raise self._error(self._peek(), 'expression in go must be function call')
        else:
            return ret

    def _parse_if(self) -> If:
        ret = If(self._next())
        ret.init = self._parse_initializer()
        ret.cond = self._parse_if_cond()
        ret.body = self._parse_compound_statement()

        # no 'else' branch
        if not self._should(self._peek(), TokenType.Keyword, 'else'):
            return ret

        # parse the 'else' branch
        self._next()
        ret.branch = self._parse_if_branch()
        return ret

    def _parse_if_cond(self) -> Expression:
        with self._Control(self):
            return self._parse_expression()

    def _parse_if_branch(self) -> Optional[Union[If, CompoundStatement]]:
        if self._should(self._peek(), TokenType.Keyword, 'if'):
            return self._parse_if()
        else:
            return self._parse_compound_statement()

    def _parse_for(self) -> Union[For, ForRange]:
        tk = self._next()
        st = self.save_state()

        # try parsing as 'for-range'
        try:
            ret = self._parse_for_range(tk)
        except SyntaxError:
            ret = None

        # not working, it must be a 'for-loop'
        if ret is None:
            self.load_state(st)
            ret = self._parse_for_loop(tk)

        # parse the loop body
        ret.body = self._parse_compound_statement()
        return ret

    def _parse_for_loop(self, tk: Token) -> For:
        ret = For(tk)
        state = self.save_state()

        # try 'for {}'
        if self._should(self._peek(), TokenType.Operator, '{'):
            return ret

        # try 'for <cond> {}'
        try:
            with self._Control(self):
                ret.cond = self._parse_expression()
                self._require(self._peek(), TokenType.Operator, '{')
        except SyntaxError:
            self.load_state(state)
        else:
            return ret

        # full 3-clauses 'for' loop, parse the initial statement if any
        if not self._should(self._peek(), TokenType.Operator, ';'):
            with self._Control(self):
                ret.init = self._parse_simple_statement()

        # the first ';'
        tk = self._next()
        self._require(tk, TokenType.Operator, ';')

        # parse the conditional expression if any
        if not self._should(self._peek(), TokenType.Operator, ';'):
            with self._Control(self):
                ret.cond = self._parse_expression()

        # the second ';'
        tk = self._next()
        self._require(tk, TokenType.Operator, ';')

        # parse the post statement if any
        if not self._should(self._peek(), TokenType.Operator, '{'):
            with self._Control(self):
                ret.post = self._parse_simple_statement()

        # all done
        return ret

    def _parse_for_range(self, tk: Token) -> ForRange:
        ret = ForRange(tk)
        ret.svd = self._is_svd()

        # check for Short Variable Declaration (SVD)
        if ret.svd:
            ret.terms = self._parse_svd()
        else:
            ret.terms = self._parse_for_range_vars()

        # must have 1 or 2 terms before the "range" operator
        if 1 <= len(ret.terms) <= 2:
            self._require(self._next(), TokenType.Keyword, 'range')
        else:
            raise self._error(self._peek(), 'invalid number of range variables')

        # parse the range expression
        with self._Control(self):
            ret.expr = self._parse_expression()
            return ret

    def _parse_for_range_eq(self, ret: List[Expression]) -> List[Expression]:
        self._require(self._next(), TokenType.Operator, '=')
        return ret

    def _parse_for_range_vars(self):
        with self._Control(self):
            return self._parse_for_range_eq(self._parse_expressions())

    def _parse_defer(self) -> Defer:
        ret = Defer(self._next())
        ret.expr = self._parse_expression()

        # must be a function call
        if not ret.expr.is_call():
            raise self._error(self._peek(), 'expression in defer must be function call')
        else:
            return ret

    def _parse_select(self) -> Select:
        ret = Select(self._next())
        self._parse_case('select', SelectCase, ret.cases, self._parse_select_case)
        return ret

    def _parse_select_case(self, case: SelectCase):
        st = self.save_state()
        expr = self._parse_expression()

        # check for send expression
        if self._should(self._peek(), TokenType.Operator, '<-'):
            case.expr = self._parse_send(self._next(), expr)
            return

        # rewind back
        self.load_state(st)
        ret = SelectReceive(self._peek())

        # try parse as SVD
        try:
            ret.svd = True
            ret.terms = self._parse_svd()
        except SyntaxError:
            self.load_state(st)
            self._parse_select_case_assign(ret)

        # parse the receive expression
        ret.value = self._parse_expression()
        case.expr = ret

    def _parse_select_case_assign(self, ret: SelectReceive):
        ret.svd = False
        ret.terms = self._parse_expressions()
        self._require(self._next(), TokenType.Operator, '=')
        return ret

    def _parse_switch(self) -> Union[Switch, TypeSwitch]:
        tk = self._next()
        init = self._parse_initializer()
        state = self.save_state()

        # set the case class and parser for expression switch,
        # so we don't have to set them in the except clause
        klass = SwitchCase
        parser = self._parse_expr_switch_case

        # first let's assume it is a type switch
        ret = TypeSwitch(tk)
        ret.name = self._parse_type_switch_init()

        # then try parsing it
        try:
            self._parse_type_switch_cond(ret)
            self._ensure_type_switch()

        # syntax error, it's just a simple expression switch, rollback
        except SyntaxError:
            ret = None
            self.load_state(state)

        # no errors, it is a type switch, commit the parsing state (by doing
        # nothing) and change the case class and parser to the correct values
        else:
            klass = TypeSwitchCase
            parser = self._parse_type_switch_case

        # parse the expression outside of the catch caluse to prevent exception
        # chaining when syntax errors occurres in the switch expression
        if ret is None:
            ret = Switch(tk)
            ret.expr = self._parse_expr_switch_cond()

        # parse the cases
        ret.init = init
        self._parse_case('switch', klass, ret.cases, parser)
        return ret

    def _parse_expr_switch_cond(self) -> Optional[Expression]:
        if not self._should(self._peek(), TokenType.Operator, '{'):
            with self._Control(self):
                return self._parse_expression()

    def _parse_expr_switch_case(self, case: SwitchCase):
        case.vals = self._parse_expressions()

    def _parse_type_switch_init(self) -> Optional[Name]:
        st = self.save_state()
        tk = self._next()

        # should be in a form of '<name> :='
        if self._should(tk, TokenType.Name):
            if self._should(self._next(), TokenType.Operator, ':='):
                return Name(tk)

        # no init clause present
        self.load_state(st)
        return None

    def _parse_type_switch_cond(self, ret: TypeSwitch):
        with self._Control(self):
            ret.type = self._parse_primary()

    def _parse_type_switch_case(self, case: TypeSwitchCase):
        case.types = self._parse_types()

    def _parse_labeled(self) -> Label:
        ret = Label(self._peek())
        ret.name = self._parse_name()
        ret.body = self._parse_statement()
        return ret

    ### Language Structures --- Statements / Control Flow Transfers ###

    def _parse_goto(self) -> Goto:
        ret = Goto(self._next())
        ret.label = self._parse_name()
        return ret

    def _parse_return(self) -> Return:
        tk = self._next()
        ret = Return(tk)

        # check for empty return
        if self._is_ops(RETURN_OPERATORS):
            return ret

        # parse the expression list
        ret.vals = self._parse_expressions()
        return ret

    def _parse_label(self) -> Optional[Name]:
        if not self._should(self._peek(), TokenType.Name):
            return None
        else:
            return self._parse_name()

    def _parse_break(self) -> Break:
        ret = Break(self._next())
        ret.label = self._parse_label()
        return ret

    def _parse_continue(self) -> Continue:
        ret = Continue(self._next())
        ret.label = self._parse_label()
        return ret

    def _parse_fallthrough(self) -> Fallthrough:
        return Fallthrough(self._next())

    ### Language Structures --- Statements / Simple Statements ###

    def _parse_send(self, tk: Token, chan: Expression) -> Send:
        ret = Send(tk)
        ret.chan = chan
        ret.expr = self._parse_expression()
        return ret

    def _parse_incdec(self, tk: Token, expr: Expression) -> IncDec:
        ret = IncDec(tk)
        ret.expr = expr
        ret.incr = self._should(self._next(), TokenType.Operator, '++')
        return ret

    def _parse_assignment(self) -> Assignment:
        ret = Assignment(self._peek())
        ret.lval = self._parse_expressions()
        ret.type = self._parse_operator(ASSIGNMENT_OPERATORS)
        ret.rval = self._parse_expressions()
        return ret

    ### Language Structures --- Statements / Generic Statements ###

    def _parse_statement(self) -> Statement:
        if self._should(self._peek(), TokenType.Keyword, 'go'):
            return self._parse_go()
        elif self._should(self._peek(), TokenType.Keyword, 'if'):
            return self._parse_if()
        elif self._should(self._peek(), TokenType.Keyword, 'for'):
            return self._parse_for()
        elif self._should(self._peek(), TokenType.Keyword, 'goto'):
            return self._parse_goto()
        elif self._should(self._peek(), TokenType.Keyword, 'defer'):
            return self._parse_defer()
        elif self._should(self._peek(), TokenType.Keyword, 'break'):
            return self._parse_break()
        elif self._should(self._peek(), TokenType.Keyword, 'return'):
            return self._parse_return()
        elif self._should(self._peek(), TokenType.Keyword, 'select'):
            return self._parse_select()
        elif self._should(self._peek(), TokenType.Keyword, 'switch'):
            return self._parse_switch()
        elif self._should(self._peek(), TokenType.Keyword, 'continue'):
            return self._parse_continue()
        elif self._should(self._peek(), TokenType.Keyword, 'fallthrough'):
            return self._parse_fallthrough()
        elif self._should(self._peek(), TokenType.Operator, '{'):
            return self._parse_compound_statement()
        elif self._is_label():
            return self._parse_labeled()
        else:
            return self._parse_simple_statement()

    def _parse_simple_statement(self) -> SimpleStatement:
        tk = self._peek()
        st = self.save_state()

        # empty statement
        if self._should(tk, TokenType.Operator, ';'):
            return Empty(tk)

        # short variable declaration
        try:
            ret = InitSpec(tk)
            ret.names = self._parse_svd()
        except SyntaxError:
            self.load_state(st)
        else:
            ret.values = self._parse_expressions()
            return ret

        # try parse one expression
        st = self.save_state()
        expr = self._parse_expression()

        # pure expression
        if self._should(self._peek(), TokenType.Operator, ';'):
            return expr

        # send statement
        if self._should(self._peek(), TokenType.Operator, '<-'):
            self._next()
            return self._parse_send(tk, expr)

        # incremental or decremental
        if self._is_ops(INCDEC_OPERATORS):
            return self._parse_incdec(tk, expr)

        # must be assignment
        self.load_state(st)
        return self._parse_assignment()

    def _parse_compound_statement(self) -> CompoundStatement:
        tk = self._next()
        ret = CompoundStatement(self._require(tk, TokenType.Operator, '{'))

        # parse each statement
        while not self._should(self._peek(), TokenType.Operator, '}'):
            ret.body.append(self._parse_statement())
            self._delimiter(';')

        # skip the '}'
        self._next()
        return ret

    ### Language Structures --- Expressions ###

    def _parse_expr(self, prec: int) -> Expression:
        if prec >= len(BINARY_OPERATORS):
            return self._parse_unary()
        else:
            return self._parse_binary(prec)

    def _parse_term(self, prec: int, op: Token, val: Expression) -> Expression:
        new = Expression(self._peek())
        new.op = Operator(op)
        new.left = val
        new.right = self._parse_expr(prec)
        return new

    def _parse_unary(self) -> Expression:
        if not self._is_ops(UNARY_OPERATORS):
            ret = Expression(self._peek())
            ret.left = self._parse_primary()
            return ret
        else:
            tk = self._next()
            ret = Expression(tk)
            ret.op = Operator(tk)
            ret.left = self._parse_unary()
            return ret

    def _parse_binary(self, prec: int) -> Expression:
        ret = Expression(self._peek())
        ret.left = self._parse_expr(prec + 1)

        # check for operators of this precedence
        while self._is_ops(BINARY_OPERATORS[prec]):
            tk = self._next()
            ret = self._parse_term(prec + 1, tk, ret)

        # all done
        return ret

    def _parse_primary(self) -> Primary:
        ret = Primary(self._peek())
        ret.val = self._parse_primary_val()
        return self._parse_primary_mods(ret)

    def _parse_primary_val(self) -> Operand:
        tk = self._peek()
        node = LIT_NODES.get(tk.kind)

        # basic literals
        if node is not None:
            return node(self._next())

        # lambda functions
        elif self._should(tk, TokenType.Keyword, 'func'):
            return self._parse_lambda()

        # a type specifier followed by a '{', it's a composite literal
        # special case: name followed by a block should be parsed as a single
        # name rather than a struct initialization when in control statements
        elif self._is_literal_type() and (self.expr >= 0 or not self._is_named_type()):
            return self._parse_composite()

        # a type specifier followed by a '(', it's a type conversion
        elif self._is_conversion_type():
            return self._parse_conversion()

        # standard identifiers (which might be a package name, but
        # we cannot distinguish package names from normal identifiers
        # in the parsing phase, it will be handled by the type inferrer)
        elif self._should(tk, TokenType.Name):
            return self._parse_name()

        # nested expressions, parse with level counter
        elif self._should(tk, TokenType.Operator, '('):
            return self._parse_nested_expr()

        # otherwise it's a syntax error
        else:
            raise self._error(tk, 'operands expected, got %s' % repr(tk.value))

    def _parse_primary_mods(self, ret: Primary) -> Primary:
        tk = self._peek()
        tt, tv = tk.kind, tk.value

        # chain every modifier
        while tt == TokenType.Operator:
            if tv == '[':
                ret.mods.append(self._parse_slice())
            elif tv == '.' and not self._is_type_expr():
                ret.mods.append(self._parse_selector())
            elif tv == '(':
                ret.mods.append(self._parse_arguments())
            else:
                break

            # locate the next token
            tk = self._peek()
            tt, tv = tk.kind, tk.value

        # all done
        return ret

    def _parse_lambda(self) -> Lambda:
        ret = Lambda(self._next())
        ret.signature = self._parse_signature()

        # function body should be parsed within nested scope
        with self._Nested(self):
            ret.body = self._parse_compound_statement()
            return ret

    def _parse_composite(self) -> Composite:
        ret = Composite(self._peek())
        ret.type = self._parse_literal_type()

        # parse the composite value within nested scope
        with self._Nested(self):
            ret.value = self._parse_composite_val()
            return ret

    def _parse_composite_val(self) -> LiteralValue:
        tk = self._next()
        ret = LiteralValue(self._require(tk, TokenType.Operator, '{'))

        # parse every element
        while not self._should(self._peek(), TokenType.Operator, '}'):
            ret.items.append(self._parse_composite_elem())
            self._delimiter(',')

        # skip the '}'
        self._next()
        return ret

    def _parse_composite_elem(self) -> Element:
        tk = self._peek()
        ret = Element(tk)

        # check for literal value
        if not self._should(tk, TokenType.Operator, '{'):
            ret.value = self._parse_expression()
        else:
            ret.value = self._parse_composite_val()

        # plain item list
        if not self._should(self._peek(), TokenType.Operator, ':'):
            return ret

        # skip the ':'
        self._next()
        ret.key = ret.value

        # check for literal value
        if not self._should(self._peek(), TokenType.Operator, '{'):
            ret.value = self._parse_expression()
        else:
            ret.value = self._parse_composite_val()

        # all done
        return ret

    def _parse_conversion(self) -> Conversion:
        ret = Conversion(self._peek())
        ret.type = self._parse_type()
        ret.value = self._parse_conversion_expr()
        return ret

    def _parse_conversion_expr(self) -> Expression:
        self._require(self._peek(), TokenType.Operator, '(')
        return self._parse_nested_expr()

    def _parse_slice(self) -> Union[Index, Slice]:
        tk = self._next()
        ret = Slice(self._require(tk, TokenType.Operator, '['))

        # ':' encountered, must be a slice
        if self._should(self._peek(), TokenType.Operator, ':'):
            self._next()
            ret.pos = None

        # otherwise assume it's an index
        else:
            with self._Nested(self):
                idx = Index(tk)
                idx.expr = ret.pos = self._parse_expression()

            # then it should ends with a ']', otherwise it must be a ':'
            if not self._should(self._peek(), TokenType.Operator, ':'):
                self._require(self._next(), TokenType.Operator, ']')
                return idx

        # optional length expression
        if self._should(self._peek(), TokenType.Operator, ':'):
            self._next()
            ret.len = None

            # it might be an empty expression
            with self._Nested(self):
                if not self._should(self._peek(), TokenType.Operator, ':'):
                    ret.len = self._parse_expression()

        # optional capacity expression
        if self._should(self._peek(), TokenType.Operator, ':'):
            self._next()
            ret.cap = None

            # it might be an empty expression
            with self._Nested(self):
                if not self._should(self._peek(), TokenType.Operator, ']'):
                    ret.cap = self._parse_expression()

        # must close with a ']'
        self._require(self._next(), TokenType.Operator, ']')
        return ret

    def _parse_selector(self) -> Union[Selector, Assertion]:
        tk = self._next()
        tk = self._require(tk, TokenType.Operator, '.')

        # might be type assertion
        if not self._should(self._peek(), TokenType.Operator, '('):
            ret = Selector(tk)
            ret.attr = self._parse_name()
        else:
            ret = Assertion(tk)
            ret.type = self._parse_nested_type()

        # all done
        return ret

    def _parse_arguments(self) -> Arguments:
        tk = self._next()
        ret = Arguments(self._require(tk, TokenType.Operator, '('))

        # empty invocation
        if self._should(self._peek(), TokenType.Operator, ')'):
            self._next()
            return ret

        # special case of the first argument, but there is a corner case:
        # in expression `foo(name, 1, 2)`, it is not possible to determain
        # whether `name` is a type name or a variable or a constant during the
        # parsing stage
        #
        # this can only happen for a few functions in real life, like `make`
        # and `new`, but since we can assign completely irrelevant values to
        # them, it is not possible to tell whether they are the original one
        # or not
        #
        # so we just simply parse them as `type specifier`, the type inferrer
        # will take care of this situation
        with self._Nested(self):
            if self._is_argument_type():
                ret.args.append(self._parse_type())
            else:
                ret.args.append(self._parse_expression())

        # have more arguments
        while self._should(self._peek(), TokenType.Operator, ','):
            self._next()
            self._parse_argument_item(self._peek(), ret.args)

        # variadic invoking
        if self._should(self._peek(), TokenType.Operator, '...'):
            self._next()
            ret.var = True

            # optional tail comma
            if self._should(self._peek(), TokenType.Operator, ','):
                self._next()

        # must end with a ')'
        self._require(self._next(), TokenType.Operator, ')')
        return ret

    def _parse_argument_item(self, tk: Token, args: List[Expression]):
        with self._Nested(self):
            if not self._should(tk, TokenType.Operator, ')'):
                args.append(self._parse_expression())

    def _parse_expression(self) -> Expression:
        return self._prune_expression_tree(self._parse_expr(prec = 0))

    ### Top Level Parsers --- Functions & Methods ###

    def _parse_function(self, ret: List[Function]):
        func = self._parse_function_def()
        func.opts, self.fflags = self.fflags, FunctionOptions(0)

        # 'go:noescape' must not have a function body
        if (func.opts & FunctionOptions.NO_ESCAPE) and \
           self._should(self._peek(), TokenType.Operator, '{'):
            raise self._error(self._peek(), 'can only use //go:noescape with external func implementations')

        # otherwise, the body is optional
        func.body = self._parse_function_body()
        ret.append(func)

    def _parse_function_def(self) -> Function:
        tk = self._peek()
        ret = Function(tk)

        # check for function receiver
        if self._should(tk, TokenType.Operator, '('):
            _, args = self._parse_parameters(for_args = False)
            ret.recv = self._ensure_receiver_args(args)

        # function name and signature type
        ret.name = self._parse_name()
        ret.type = self._parse_signature()
        return ret

    def _parse_function_body(self) -> Optional[CompoundStatement]:
        with self._Nested(self):
            if not self._should(self._peek(), TokenType.Operator, '{'):
                return None
            else:
                return self._parse_compound_statement()

    ### Top Level Parsers --- Variables, Types, Constants & Imports ###

    def _parse_var_spec(self, tk: Token, ret: List[InitSpec], consts: bool):
        val = InitSpec(tk)
        val.names.append(Name(self._require(tk, TokenType.Name)))

        # const a, b, c, ...
        while self._should(self._peek(), TokenType.Operator, ','):
            self._next()
            val.names.append(self._parse_name())

        # optional const type, an approximated guessing is good enough
        if self._is_probably_type():
            val.type = self._parse_type()

        # the '=' operator, required under read-only mode for the first item
        if (consts and self.iota == 0) or self._should(self._peek(), TokenType.Operator, '='):
            self._require(self._next(), TokenType.Operator, '=')
            val.values = self._parse_expressions()

        # bare identifiers for remaining consts, just copy the previous expressions
        elif consts and self.iota > 0:
            assert ret
            val.values = ret[-1].values[:]

        # otherwise a type declaration is required
        elif not val.type:
            raise self._error(self._peek(), '\'=\' or type specifier expected')

        # add to variable / const list
        val.consts = consts
        ret.append(val)

    def _parse_type_spec(self, tk: Token, ret: List[TypeSpec]):
        ts = TypeSpec(tk)
        ts.name = Name(self._require(tk, TokenType.Name))

        # check for type aliasing
        if self._should(self._peek(), TokenType.Operator, '='):
            self._next()
            ts.alias = True

        # parse the actual type
        ts.type = self._parse_type()
        ret.append(ts)

    def _parse_import_spec(self, tk: Token, ret: List[ImportSpec]):
        imp = ImportSpec(tk)
        tt, tv = tk.kind, tk.value

        # import xxx 'xxx'
        if tt == TokenType.Name:
            imp.alias = Name(tk)
            tk = self._next()

        # import . 'xxx'
        elif tt == TokenType.Operator and tv == '.':
            imp.alias = ImportHere(tk)
            tk = self._next()

        # add to import list
        imp.path = String(self._require(tk, TokenType.String))
        ret.append(imp)

    ### Groupped Parsers ###

    def _group_reset(self):
        self.iota = 0
        self.fflags = FunctionOptions(0)

    def _group_increment(self):
        self.iota += 1
        self.fflags = FunctionOptions(0)

    def _parse_declarations(
        self,
        ret    : List[Union[InitSpec, TypeSpec, ImportSpec]],
        parser : Callable[[Token, List[Union[InitSpec, TypeSpec, ImportSpec]]], None],
    ):
        if not self._should(self._peek(), TokenType.Operator, '('):
            self._group_reset()
            parser(self._next(), ret)
        else:
            self._next()
            self._group_reset()
            self._parse_declaration_group(ret, parser)
            self._require(self._next(), TokenType.Operator, ')')

    def _parse_declaration_group(
        self,
        ret    : List[Union[InitSpec, TypeSpec, ImportSpec]],
        parser : Callable[[Token, List[Union[InitSpec, TypeSpec, ImportSpec]]], None],
    ):
        while not self._should(self._peek(), TokenType.Operator, ')'):
            parser(self._next(), ret)
            self._group_increment()
            self._delimiter(';')

    ### Generic Parsers ###

    def _parse_dir(self, tk: Token, ret: Package):
        if isinstance(tk.value, NoSplitDirective):
            self.fflags |= FunctionOptions.NO_SPLIT
        elif isinstance(tk.value, NoEscapeDirective):
            self.fflags |= FunctionOptions.NO_ESCAPE
        elif isinstance(tk.value, LinkNameDirective):
            ls = LinkSpec(tk)
            ls.name = tk.value.name
            ls.link = tk.value.link
            ret.links.append(ls)
        else:
            raise SystemError('invalid compiler directive')

    def _parse_val(self, tk: Token, ret: List[InitSpec]):
        self._parse_var_spec(tk, ret, consts= True)

    def _parse_var(self, tk: Token, ret: List[InitSpec]):
        self._parse_var_spec(tk, ret, consts= False)

    def _parse_decl(self, tk: Token, ret: Package):
        if tk.value == 'func':
            self._parse_function(ret.funcs)
        elif tk.value == 'var':
            self._parse_declarations(ret.vars, self._parse_var)
        elif tk.value == 'type':
            self._parse_declarations(ret.types, self._parse_type_spec)
        elif tk.value == 'const':
            self._parse_declarations(ret.consts, self._parse_val)
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
        self._delimiter(';')

        # imports go before other declarations
        while self._should(self._peek(), TokenType.Keyword, 'import'):
            self._next()
            self._parse_declarations(ret.imports, self._parse_import_spec)
            self._delimiter(';')

        # parse other top-level declarations
        while True:
            if self._should(self._peek(), TokenType.Directive):
                self._parse_dir(self._next(), ret)
            elif self._should(self._peek(), TokenType.Keyword):
                self._parse_decl(self._next(), ret)
                self._delimiter(';')
            else:
                break

        # must be EOF
        if self._should(self._peek(), TokenType.End):
            return ret

        # otherwise it's an unexpected token
        tk = self._next()
        raise self._error(tk, 'unexpected token %s' % repr(tk))
