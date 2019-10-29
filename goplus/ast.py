# -*- coding: utf-8 -*-

import json
import inspect

from typing import Any
from typing import Set
from typing import cast
from typing import Dict
from typing import List
from typing import Union
from typing import Optional
from typing import Sequence

from .types import Type as T
from .utils import StrictFields
from .flags import ChannelOptions
from .flags import FunctionOptions

from .tokenizer import Token
from .tokenizer import TokenType
from .tokenizer import TokenValue

class Node(metaclass = StrictFields):
    vt   : Optional[T]
    row  : int
    col  : int
    file : str

    # don't initialize these fields in the generated constructor
    __noinit__ = {
        'row',
        'col',
        'file',
    }

    def __init__(self, tk: Token):
        self.row = tk.row
        self.col = tk.col
        self.file = tk.file

    def __repr__(self) -> str:
        return json.dumps(self._build(set()), indent = 4)

    def _build(self, path: Set[int]) -> Dict[str, Any]:
        if id(self) in path:
            return {}

        # add to path
        ret = {'__class__': self.__class__.__name__}
        path.add(id(self))

        # dump every exported attrs, except "row", "col" and "file"
        for attr in dir(self):
            if attr == 'row' or \
               attr == 'col' or \
               attr == 'file' or \
               attr.endswith('_') or \
               attr.startswith('_') or \
               inspect.ismethod(getattr(self, attr)):
                continue
            elif attr != 'vt':
                ret[attr] = self._build_val(path, getattr(self, attr))
            elif self.vt is not None:
                ret[attr] = str(self.vt)
            else:
                ret[attr] = None

        # all done
        path.remove(id(self))
        return ret

    def _build_val(self, path: Set[int], val: Any) -> Any:
        if isinstance(val, Node):
            return val._build(path)
        elif isinstance(val, complex):
            return str(val)
        elif isinstance(val, bytes):
            return val.decode('unicode_escape')
        elif isinstance(val, dict):
            return self._build_dict(path, val)
        elif isinstance(val, (list, tuple)):
            return self._build_list(path, val)
        else:
            return val

    def _build_list(self, path: Set[int], val: Sequence[Any]) -> List[Any]:
        return [self._build_val(path, item) for item in val]

    def _build_dict(self, path: Set[int], val: Dict[Any, Any]) -> Dict[Any, Any]:
        return {key: self._build_val(path, value) for key, value in val.items()}

    def clone(self) -> 'Node':
        ret = self.__class__.__new__(self.__class__)
        ret.vt = self.vt
        ret.col = self.col
        ret.row = self.row
        ret.file = self.file
        return ret

### Basic Elements ###

class Value(Node):
    kind  : TokenType
    value : TokenValue

    def __init__(self, tk: Token):
        super().__init__(tk)
        self.value = tk.value
        assert tk.kind == self.kind

    def clone(self) -> 'Value':
        return self

class Int(Value):
    kind = TokenType.Int

# virtual AST node, used by type inferrer,
# thus the parser never yields this AST node
class Nil(Value):
    kind = TokenType.Nil

# virtual AST node, used by type inferrer,
# thus the parser never yields this AST node
class Bool(Value):
    kind = TokenType.Bool

class Name(Value):
    kind = TokenType.Name

class Rune(Value):
    kind = TokenType.Rune

class Float(Value):
    kind = TokenType.Float

class String(Value):
    kind = TokenType.String

class Complex(Value):
    kind = TokenType.Complex

class Operator(Value):
    kind = TokenType.Operator

### Language Structures ###

class MapType(Node):
    key  : 'Type'
    elem : 'Type'

    def clone(self) -> 'MapType':
        ret = cast(MapType, super().clone())
        ret.key = self.key.clone()
        ret.elem = self.elem.clone()
        return ret

class ArrayType(Node):
    len  : 'Expression'
    elem : 'Type'

    def clone(self) -> 'ArrayType':
        ret = cast(ArrayType, super().clone())
        ret.len = self.len.clone()
        ret.elem = self.elem.clone()
        return ret

class SliceType(Node):
    elem: 'Type'

    def clone(self) -> 'SliceType':
        ret = cast(SliceType, super().clone())
        ret.elem = self.elem.clone()
        return ret

class NamedType(Node):
    name    : Name
    package : Optional[Name]

    def clone(self) -> 'NamedType':
        ret = cast(NamedType, super().clone())
        ret.name = self.name.clone()
        ret.package = self.package and self.package.clone()
        return ret

class StructType(Node):
    fields: List['StructField']

    def clone(self) -> 'StructType':
        ret = cast(StructType, super().clone())
        ret.fields = [v.clone() for v in self.fields]
        return ret

class StructField(Node):
    type: 'Type'
    name: Optional[Name]
    tags: Optional[String]

    def clone(self) -> 'StructField':
        ret = cast(StructField, super().clone())
        ret.type = self.type.clone()
        ret.name = self.name and self.name.clone()
        ret.tags = self.tags and self.tags.clone()
        return ret

class ChannelType(Node):
    dir  : ChannelOptions
    elem : 'Type'

    def clone(self) -> 'ChannelType':
        ret = cast(ChannelType, super().clone())
        ret.dir = self.dir
        ret.elem = self.elem.clone()
        return ret

class PointerType(Node):
    base: 'Type'

    def clone(self) -> 'PointerType':
        ret = cast(PointerType, super().clone())
        ret.base = self.base.clone()
        return ret

class FunctionType(Node):
    type: 'FunctionSignature'

    def clone(self) -> 'FunctionType':
        ret = cast(FunctionType, super().clone())
        ret.type = self.type.clone()
        return ret

class FunctionArgument(Node):
    name: Optional[Name]
    type: 'Type'

    def clone(self) -> 'FunctionArgument':
        ret = cast(FunctionArgument, super().clone())
        ret.name = self.name and self.name.clone()
        ret.type = self.type.clone()
        return ret

class FunctionSignature(Node):
    var  : bool
    args : List[FunctionArgument]
    rets : List[FunctionArgument]

    def clone(self) -> 'FunctionSignature':
        ret = cast(FunctionSignature, super().clone())
        ret.var = self.var
        ret.args = [v.clone() for v in self.args]
        ret.rets = [v.clone() for v in self.rets]
        return ret

class InterfaceType(Node):
    decls: List[Union[
        NamedType,
        'InterfaceMethod',
    ]]

    def clone(self) -> 'InterfaceType':
        ret = cast(InterfaceType, super().clone())
        ret.decls = [v.clone() for v in self.decls]
        return ret

class InterfaceMethod(Node):
    name: Name
    type: FunctionSignature

    def clone(self) -> 'InterfaceMethod':
        ret = cast(InterfaceMethod, super().clone())
        ret.name = self.name.clone()
        ret.type = self.type.clone()
        return ret

Type = Union[
    MapType,
    ArrayType,
    SliceType,
    NamedType,
    StructType,
    ChannelType,
    PointerType,
    FunctionType,
    InterfaceType,
]

class Primary(Node):
    val  : 'Operand'
    mods : List['Modifier']

    def clone(self) -> 'Primary':
        ret = cast(Primary, super().clone())
        ret.val = self.val.clone()
        ret.mods = [v.clone() for v in self.mods]
        return ret

    def is_index(self) -> bool:
        if not self.mods:
            return False
        else:
            return isinstance(self.mods[-1], Index)

    def is_selector(self) -> bool:
        if not self.mods:
            return False
        else:
            return isinstance(self.mods[-1], Selector)

class Conversion(Node):
    type  : Type
    value : 'Expression'

    def clone(self) -> 'Conversion':
        ret = cast(Conversion, super().clone())
        ret.type = self.type.clone()
        ret.value = self.value.clone()
        return ret

class Expression(Node):
    op    : Optional[Operator]
    left  : Union[Primary, 'Expression']
    right : Optional['Expression']

    def clone(self) -> 'Expression':
        ret = cast(Expression, super().clone())
        ret.op = self.op and self.op.clone()
        ret.left = self.left.clone()
        ret.right = self.right and self.right.clone()
        return ret

    def is_call(self) -> bool:
        if self.op is not None:
            return False
        elif self.right is not None:
            return False
        elif not isinstance(self.left, Primary):
            return False
        elif not self.left.mods:
            return False
        else:
            return isinstance(self.left.mods[-1], Arguments)

class Lambda(Node):
    body      : 'CompoundStatement'
    signature : FunctionSignature

    def clone(self) -> 'Lambda':
        ret = cast(Lambda, super().clone())
        ret.body = self.body.clone()
        ret.signature = self.signature.clone()
        return ret

class VarArrayType(Node):
    elem: Type

    def clone(self) -> 'VarArrayType':
        ret = cast(VarArrayType, super().clone())
        ret.elem = self.elem.clone()
        return ret

LiteralType = Union[
    MapType,
    ArrayType,
    NamedType,
    SliceType,
    StructType,
    VarArrayType,
]

class LiteralValue(Node):
    items: List['Element']

    def clone(self) -> 'LiteralValue':
        ret = cast(LiteralValue, super().clone())
        ret.items = [v.clone() for v in self.items]
        return ret

class Element(Node):
    key   : Optional[Union[Expression, LiteralValue]]
    value : Union[Expression, LiteralValue]

    def clone(self) -> 'Element':
        ret = cast(Element, super().clone())
        ret.key = self.key and self.key.clone()
        ret.value = self.value.clone()
        return ret

class Composite(Node):
    type  : LiteralType
    value : LiteralValue

    def clone(self) -> 'Composite':
        ret = cast(Composite, super().clone())
        ret.type = self.type.clone()
        ret.value = self.value.clone()
        return ret

Constant = Union[
    Int,
    Bool,
    Rune,
    Float,
    String,
    Complex,
]

Operand = Union[
    Name,
    Lambda,
    Constant,
    Composite,
    Conversion,
    Expression,
]

class Index(Node):
    expr: Expression

    def clone(self) -> 'Index':
        ret = cast(Index, super().clone())
        ret.expr = self.expr.clone()
        return ret

class Slice(Node):
    pos: Optional[Expression]
    len: Optional[Expression]
    cap: Optional[Union[bool, Expression]]

    def clone(self) -> 'Slice':
        ret = cast(Slice, super().clone())
        ret.pos = self.pos and self.pos.clone()
        ret.len = self.len and self.len.clone()
        ret.cap = self.cap and self.cap.clone()
        return ret

class Selector(Node):
    attr: Name

    def clone(self) -> 'Selector':
        ret = cast(Selector, super().clone())
        ret.attr = self.attr.clone()
        return ret

class Arguments(Node):
    var  : bool
    args : List[Union[Type, Expression]]

    def clone(self) -> 'Arguments':
        ret = cast(Arguments, super().clone())
        ret.var = self.var
        ret.args = [v.clone() for v in self.args]
        return ret

class Assertion(Node):
    type: Optional[Type]

    def clone(self) -> 'Assertion':
        ret = cast(Assertion, super().clone())
        ret.type = self.type and self.type.clone()
        return ret

Modifier = Union[
    Index,
    Slice,
    Selector,
    Arguments,
    Assertion,
]

### Top Level Declarations ###

class LinkSpec(Node):
    name: str
    link: str

    def clone(self) -> 'LinkSpec':
        return self

class InitSpec(Node):
    iota   : Optional[int]
    type   : Optional[Type]
    names  : List[Name]
    values : List[Expression]
    consts : bool

    def clone(self) -> 'InitSpec':
        ret = cast(InitSpec, super().clone())
        ret.iota = self.iota
        ret.type = self.type and self.type.clone()
        ret.names = [v.clone() for v in self.names]
        ret.values = [v.clone() for v in self.values]
        ret.consts = self.consts
        return ret

class TypeSpec(Node):
    name  : Name
    type  : 'Type'
    alias : bool

    def clone(self) -> 'TypeSpec':
        ret = cast(TypeSpec, super().clone())
        ret.name = self.name.clone()
        ret.type = self.type.clone()
        ret.alias = self.alias
        return ret

class Function(Node):
    name: Name
    opts: FunctionOptions
    type: FunctionSignature
    recv: Optional[FunctionArgument]
    body: Optional['CompoundStatement']

    def clone(self) -> 'Function':
        ret = cast(Function, super().clone())
        ret.opts = self.opts
        ret.name = self.name.clone()
        ret.type = self.type.clone()
        ret.recv = self.recv and self.recv.clone()
        ret.body = self.body and self.body.clone()
        return ret

class ImportC(Node):
    src: str

    def __init__(self, tk: Token):
        self.src = tk.value
        super().__init__(tk)
        assert tk.kind == TokenType.Comments

    def clone(self) -> 'ImportC':
        return self

class ImportHere(Node):
    def __init__(self, tk: Token):
        super().__init__(tk)
        assert tk.kind == TokenType.Operator and tk.value == '.'

    def clone(self) -> 'ImportHere':
        return self

class ImportSpec(Node):
    path  : String
    alias : Optional[Union[Name, ImportC, ImportHere]]

    def clone(self) -> 'ImportSpec':
        ret = cast(ImportSpec, super().clone())
        ret.path = self.path.clone()
        ret.alias = self.alias and self.alias.clone()
        return ret

class Package(Node):
    name    : Name
    vars    : List[InitSpec]
    links   : List[LinkSpec]
    funcs   : List[Function]
    types   : List[TypeSpec]
    consts  : List[InitSpec]
    imports : List[ImportSpec]

    def clone(self) -> 'ImportSpec':
        ret = cast(ImportSpec, super().clone())
        ret.name = self.name.clone()
        ret.vars = [v.clone() for v in self.vars]
        ret.links = [v.clone() for v in self.links]
        ret.funcs = [v.clone() for v in self.funcs]
        ret.types = [v.clone() for v in self.types]
        ret.consts = [v.clone() for v in self.consts]
        ret.imports = [v.clone() for v in self.imports]
        return ret

### Statements -- Basic Structures ###

class Go(Node):
    expr: Expression

    def clone(self) -> 'Go':
        ret = cast(Go, super().clone())
        ret.expr = self.expr.clone()
        return ret

class If(Node):
    cond   : Expression
    init   : 'SimpleStatement'
    body   : 'CompoundStatement'
    branch : Optional[Union['If', 'CompoundStatement']]

    def clone(self) -> 'If':
        ret = cast(If, super().clone())
        ret.cond = self.cond.clone()
        ret.init = self.init.clone()
        ret.body = self.body.clone()
        ret.branch = self.branch and self.branch.clone()
        return ret

class For(Node):
    cond: Optional[Expression]
    init: Optional['SimpleStatement']
    post: Optional['SimpleStatement']
    body: 'CompoundStatement'

    def clone(self) -> 'For':
        ret = cast(For, super().clone())
        ret.body = self.body.clone()
        ret.cond = self.cond and self.cond.clone()
        ret.init = self.init and self.init.clone()
        ret.post = self.post and self.post.clone()
        return ret

class ForRange(Node):
    svd   : bool
    expr  : Expression
    body  : 'CompoundStatement'
    terms : List[Union[Name, Expression]]

    def clone(self) -> 'ForRange':
        ret = cast(ForRange, super().clone())
        ret.svd = self.svd
        ret.expr = self.expr.clone()
        ret.body = self.body.clone()
        ret.terms = [v.clone() for v in self.terms]
        return ret

class Defer(Node):
    expr: Expression

    def clone(self) -> 'Defer':
        ret = cast(Defer, super().clone())
        ret.expr = self.expr.clone()
        return ret

class Select(Node):
    cases: List['SelectCase']

    def clone(self) -> 'Select':
        ret = cast(Select, super().clone())
        ret.cases = [v.clone() for v in self.cases]
        return ret

class SelectCase(Node):
    body: List['Statement']
    expr: Optional[Union['Send', 'SelectReceive']]

    def clone(self) -> 'SelectCase':
        ret = cast(SelectCase, super().clone())
        ret.body = [v.clone() for v in self.body]
        ret.expr = self.expr and self.expr.clone()
        return ret

class SelectReceive(Node):
    svd   : bool
    value : Expression
    terms : List[Union[Name, Expression]]

    def clone(self) -> 'SelectReceive':
        ret = cast(SelectReceive, super().clone())
        ret.svd = self.svd
        ret.value = self.value.clone()
        ret.terms = [v.clone() for v in self.terms]
        return ret

class Switch(Node):
    expr  : Optional[Expression]
    init  : Optional['SimpleStatement']
    cases : List['SwitchCase']

    def clone(self) -> 'Switch':
        ret = cast(Switch, super().clone())
        ret.expr = self.expr and self.expr.clone()
        ret.init = self.init and self.init.clone()
        ret.cases = [v.clone() for v in self.cases]
        return ret

class SwitchCase(Node):
    vals: List[Expression]
    body: List['Statement']

    def clone(self) -> 'Switch':
        ret = cast(Switch, super().clone())
        ret.vals = [v.clone() for v in self.vals]
        ret.body = [v.clone() for v in self.body]
        return ret

class TypeSwitch(Node):
    name  : Optional[Name]
    type  : Optional[Primary]
    init  : Optional['SimpleStatement']
    cases : List['TypeSwitchCase']

    def clone(self) -> 'TypeSwitch':
        ret = cast(TypeSwitch, super().clone())
        ret.name = self.name and self.name.clone()
        ret.type = self.type and self.type.clone()
        ret.init = self.init and self.init.clone()
        ret.cases = [v.clone() for v in self.cases]
        return ret

class TypeSwitchCase(Node):
    body  : List['Statement']
    types : List[Type]

    def clone(self) -> 'TypeSwitchCase':
        ret = cast(TypeSwitchCase, super().clone())
        ret.body = [v.clone() for v in self.body]
        ret.types = [v.clone() for v in self.types]
        return ret

### Statements -- Control Flow Transfers ###

class Goto(Node):
    label: Name

    def clone(self) -> 'Goto':
        ret = cast(Goto, super().clone())
        ret.label = self.label.clone()
        return ret

class Label(Node):
    name: Name
    body: 'Statement'

    def clone(self) -> 'Label':
        ret = cast(Label, super().clone())
        ret.name = self.name.clone()
        ret.body = self.body.clone()
        return ret

class Return(Node):
    vals: List[Expression]

    def clone(self) -> 'Return':
        ret = cast(Return, super().clone())
        ret.vals = [v.clone() for v in self.vals]
        return ret

class Break(Node):
    label: Optional[Name]

    def clone(self) -> 'Break':
        ret = cast(Break, super().clone())
        ret.label = self.label and self.label.clone()
        return ret

class Continue(Node):
    label: Optional[Name]

    def clone(self) -> 'Continue':
        ret = cast(Continue, super().clone())
        ret.label = self.label and self.label.clone()
        return ret

class Fallthrough(Node):
    def clone(self) -> 'Fallthrough':
        return self

### Statements -- Simple Statements ###

class Send(Node):
    chan: Expression
    expr: Expression

    def clone(self) -> 'Send':
        ret = cast(Send, super().clone())
        ret.chan = self.chan.clone()
        ret.expr = self.expr.clone()
        return ret

class Empty(Node):
    def clone(self) -> 'Empty':
        return self

class IncDec(Node):
    incr: bool
    expr: Expression

    def clone(self) -> 'IncDec':
        ret = cast(IncDec, super().clone())
        ret.incr = self.incr
        ret.expr = self.expr.clone()
        return ret

class Assignment(Node):
    type: Operator
    lval: List[Expression]
    rval: List[Expression]

    def clone(self) -> 'Assignment':
        ret = cast(Assignment, super().clone())
        ret.type = self.type.clone()
        ret.lval = [v.clone() for v in self.lval]
        ret.rval = [v.clone() for v in self.rval]
        return ret

SimpleStatement = Union[
    Send,
    Empty,
    IncDec,
    InitSpec,
    Assignment,
    Expression,
]

### Statements -- Generic Statements ###

class CompoundStatement(Node):
    body: List['Statement']

    def clone(self) -> 'CompoundStatement':
        ret = cast(CompoundStatement, super().clone())
        ret.body = [v.clone() for v in self.body]
        return ret

Statement = Union[
    Go,
    If,
    For,
    Goto,
    Break,
    Defer,
    Label,
    Switch,
    Select,
    Return,
    ForRange,
    Continue,
    TypeSwitch,
    Fallthrough,
    List[InitSpec],
    List[TypeSpec],
    SimpleStatement,
    CompoundStatement,
]
