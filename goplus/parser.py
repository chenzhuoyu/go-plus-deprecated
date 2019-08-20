# -*- coding: utf-8 -*-

from typing import Set
from typing import List
from typing import Tuple
from typing import Union
from typing import TypeVar
from typing import Callable
from typing import Optional

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
from .ast import ChannelDirection

from .ast import StructField
from .ast import InterfaceMethod

from .ast import FunctionArgument
from .ast import FunctionSignature

from .tokenizer import State
from .tokenizer import Token
from .tokenizer import Tokenizer

from .tokenizer import TokenType
from .tokenizer import TokenValue

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

ARGUMENT_OPERATORS = {
    ',',
    ')',
    '...',
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

    class _Scope:
        ps: 'Parser'
        st: Optional[Tuple[State, Token, Token, int]]

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

    def __init__(self, lx: Tokenizer):
        self.lx = lx
        self.iota = 0
        self.last = None
        self.prev = None

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
            raise self._error(tk, '%s, \')\', \'}\' or new line expected' % repr(delim))

        # skip the actual delimiter
        if tv == delim:
            self._next()

    ### State Management ###

    def save_state(self) -> Tuple[State, Token, Token, int]:
        return self.lx.save_state(), self.last, self.prev, self.iota

    def load_state(self, state: Tuple[State, Token, Token, int]):
        st, self.last, self.prev, self.iota = state
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

    def _is_ops(self, ops: Set[str]) -> bool:
        tk = self._peek()
        return tk.value in ops and tk.kind == TokenType.Operator

    def _is_literal_type(self) -> bool:
        with self._Scope(self):
            try:
                self._parse_literal_type()
            except SyntaxError:
                return False
            else:
                return self._should(self._next(), TokenType.Operator, '{')

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
                self._parse_type()
            except SyntaxError:
                return False
            else:
                return self._should(self._next(), TokenType.Operator, '(')

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

    ### Basic Name Parser ###

    def _parse_name(self) -> Name:
        return Name(self._require(self._next(), TokenType.Name))

    ### Nested Parsers ###

    def _parse_nested_end(self, ret: Union[Type, Expression]) -> Union[Type, Expression]:
        self._require(self._next(), TokenType.Operator, ')')
        return ret

    def _parse_nested_type(self) -> Type:
        self._next()
        return self._parse_nested_end(self._parse_type())

    def _parse_nested_expr(self) -> Expression:
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
        ret.key = self._parse_type()
        self._require(self._next(), TokenType.Operator, ']')
        ret.elem = self._parse_type()
        return ret

    def _parse_list_type(self, for_literal: bool) -> Union[ArrayType, SliceType, VarArrayType]:
        tk = self._next()
        ret = SliceType(tk)

        # check for array length specifier
        if self._should(self._peek(), TokenType.Operator, ']'):
            self._next()
        elif self._should(self._peek(), TokenType.Operator, '...') and for_literal:
            ret = VarArrayType(tk)
            self._require(self._next(), TokenType.Operator, ']')
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
        ret = FunctionType(self._next())
        ret.signature = self._parse_signature()
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
        ret.signature = self._parse_signature()
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

        # no return values
        if self._should(self._peek(), TokenType.Operator, ';'):
            return ret

        # multiple return values
        if self._should(self._peek(), TokenType.Operator, '('):
            _, ret.rets = self._parse_parameters(for_args = False)
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

    ### Language Structures --- Statements ###

    def _parse_compound_statement(self) -> CompoundStatement:
        pass    # TODO: function body

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
        elif self._is_literal_type():
            return self._parse_composite(with_type = True)

        # a type specifier followed by a '(', it's a type conversion
        elif self._is_conversion_type():
            return self._parse_conversion()

        # standard identifiers (which might be a package name, but
        # we cannot distinguish package names from normal identifiers
        # in the parsing phase, it will be handled by the type inferrer)
        elif self._should(tk, TokenType.Name):
            return self._parse_name()

        # nested expressions
        elif self._should(tk, TokenType.Operator, '('):
            return self._parse_nested_expr()

        # composite literals without literal type
        elif self._should(tk, TokenType.Operator, '{'):
            return self._parse_composite(with_type = False)

        # otherwise it's a syntax error
        else:
            raise self._error(tk, 'operands expected, got %s' % repr(tk))

    def _parse_primary_mods(self, ret: Primary) -> Primary:
        tk = self._peek()
        tt, tv = tk.kind, tk.value

        # chain every modifier
        while tt == TokenType.Operator:
            if tv == '[':
                ret.mods.append(self._parse_slice())
            elif tv == '.':
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
        ret.body = self._parse_compound_statement()
        return ret

    def _parse_composite(self, with_type: bool) -> Composite:
        ret = Composite(self._peek())
        ret.type = self._parse_literal_type() if with_type else None
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
        if not self._should(tk, TokenType.Operator, '{'):
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
            if not self._should(self._peek(), TokenType.Operator, ':'):
                ret.len = self._parse_expression()

        # optional capacity expression
        if self._should(self._peek(), TokenType.Operator, ':'):
            self._next()
            ret.cap = None

            # it might be an empty expression
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
        if not self._should(tk, TokenType.Operator, ')'):
            args.append(self._parse_expression())

    def _parse_expression(self) -> Expression:
        return self._prune_expression_tree(self._parse_expr(prec = 0))

    ### Top Level Parsers --- Functions & Methods ###

    def _parse_function(self, ret: List[Function]):
        tk = self._peek()
        func = Function(tk)

        # check for function receiver
        if self._should(tk, TokenType.Operator, '('):
            _, args = self._parse_parameters(for_args = False)
            func.receiver = self._ensure_receiver_args(args)

        # parse the signature and function body
        func.name = self._parse_name()
        func.signature = self._parse_signature()
        func.body = self._parse_compound_statement()
        ret.append(func)

    ### Top Level Parsers --- Variables, Types, Constants & Imports ###

    def _parse_var_spec(self, tk: Token, ret: List[InitSpec], readonly: bool):
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
        if (readonly and self.iota == 0) or self._should(self._peek(), TokenType.Operator, '='):
            self._require(self._next(), TokenType.Operator, '=')
            val.values.append(self._parse_expression())

            # ... = 1, 2, 3, ...
            while self._should(self._peek(), TokenType.Operator, ','):
                self._next()
                val.values.append(self._parse_expression())

        # bare identifiers for remaining consts, just copy the previous expressions
        elif readonly and self.iota > 0:
            assert ret
            val.values = ret[-1].values[:]

        # otherwise a type declaration is required
        elif not val.type:
            raise self._error(self._peek(), '\'=\' or type specifier expected')

        # add to variable / const list
        val.readonly = readonly
        ret.append(val)

    def _parse_type_spec(self, tk: Token, ret: List[TypeSpec]):
        ts = TypeSpec(tk)
        ts.name = Name(self._require(tk, TokenType.Name))

        # check for type aliasing
        if self._should(self._peek(), TokenType.Operator, '='):
            self._next()
            ts.is_alias = True

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
            self._delimiter(';')

    ### Generic Parsers ###

    def _parse_val(self, tk: Token, ret: List[InitSpec]):
        self._parse_var_spec(tk, ret, readonly = True)

    def _parse_var(self, tk: Token, ret: List[InitSpec]):
        self._parse_var_spec(tk, ret, readonly = False)

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
        while self._should(self._peek(), TokenType.Keyword):
            self._parse_decl(self._next(), ret)
            self._delimiter(';')

        # must be EOF
        if self._should(self._peek(), TokenType.End):
            return ret

        # otherwise it's an unexpected token
        tk = self._next()
        raise self._error(tk, 'unexpected token %s' % repr(tk))
