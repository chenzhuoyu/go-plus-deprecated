# -*- coding: utf-8 -*-

import json
import inspect

from enum import IntFlag
from .utils import AnnotationSlots

from typing import Any
from typing import Set
from typing import Dict
from typing import List
from typing import Union
from typing import Optional
from typing import Sequence

from .tokenizer import Token
from .tokenizer import TokenType
from .tokenizer import TokenValue

class Node(metaclass = AnnotationSlots):
    row  : int
    col  : int
    file : str

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
            if attr != 'row' and \
               attr != 'col' and \
               attr != 'file' and \
               not attr.endswith('_') and \
               not attr.startswith('_') and \
               not inspect.ismethod(getattr(self, attr)):
                ret[attr] = self._build_val(path, getattr(self, attr))

        # all done
        path.remove(id(self))
        return ret

    def _build_val(self, path: Set[int], val: Any) -> Any:
        if isinstance(val, Node):
            return val._build(path)
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

### Basic Elements ###

class Value(Node):
    kind  : TokenType
    value : TokenValue

    def __init__(self, tk: Token):
        super().__init__(tk)
        self.value = tk.value
        assert tk.kind == self.kind

class Int(Value):
    kind = TokenType.Int

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

class ArrayType(Node):
    len  : 'Expression'
    elem : 'Type'

class SliceType(Node):
    elem: 'Type'

class NamedType(Node):
    name    : Name
    package : Optional[Name]

    def __init__(self, tk: Token):
        self.package = None
        super().__init__(tk)

class StructType(Node):
    fields: List['StructField']

    def __init__(self, tk: Token):
        self.fields = []
        super().__init__(tk)

class StructField(Node):
    type: 'Type'
    name: Optional[Name]
    tags: Optional[String]

    def __init__(self, tk: Token):
        self.name = None
        self.tags = None
        super().__init__(tk)

class ChannelDirection(IntFlag):
    Send = 0x01
    Recv = 0x02
    Both = Send | Recv

class ChannelType(Node):
    dir  : ChannelDirection
    elem : 'Type'

    def __init__(self, tk: Token):
        super().__init__(tk)
        self.dir = ChannelDirection.Both

class PointerType(Node):
    base: 'Type'

class FunctionType(Node):
    signature: 'FunctionSignature'

class FunctionArgument(Node):
    name: Optional[Name]
    type: 'Type'

    def __init__(self, tk: Token):
        self.name = None
        super().__init__(tk)

class FunctionSignature(Node):
    var  : bool
    args : List[FunctionArgument]
    rets : List[FunctionArgument]

    def __init__(self, tk: Token):
        self.var = False
        self.args = []
        self.rets = []
        super().__init__(tk)

class InterfaceType(Node):
    decls: List[Union[
        NamedType,
        'InterfaceMethod',
    ]]

    def __init__(self, tk: Token):
        self.decls = []
        super().__init__(tk)

class InterfaceMethod(Node):
    name      : Name
    signature : FunctionSignature

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

    def __init__(self, tk: Token):
        self.mods = []
        super().__init__(tk)

class Conversion(Node):
    type  : Type
    value : 'Expression'

class Expression(Node):
    op    : Optional[Operator]
    left  : Union[Primary, 'Expression']
    right : Optional['Expression']

    def __init__(self, tk: Token):
        self.op = None
        self.right = None
        super().__init__(tk)

class Lambda(Node):
    pass    # TODO: define this

class VarArrayType(Node):
    elem: Type

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

    def __init__(self, tk: Token):
        self.items = []
        super().__init__(tk)

class Element(Node):
    key   : Optional[Union[Expression, LiteralValue]]
    value : Union[Expression, LiteralValue]

    def __init__(self, tk: Token):
        self.key = None
        super().__init__(tk)

class Composite(Node):
    type  : Optional[LiteralType]
    value : LiteralValue

    def __init__(self, tk: Token):
        self.type = None
        super().__init__(tk)

Operand = Union[
    Int,
    Name,
    Rune,
    Float,
    Lambda,
    String,
    Complex,
    Composite,
    Conversion,
    Expression,
]

class Index(Node):
    expr: Expression

class Slice(Node):
    pos: Optional[Expression]
    len: Optional[Expression]
    cap: Optional[Union[bool, Expression]]

    def __init__(self, tk: Token):
        self.pos = None
        self.len = None
        self.cap = False
        super().__init__(tk)

class Selector(Node):
    attr: Name

class Arguments(Node):
    var  : bool
    args : List[Union[Type, Expression]]

    def __init__(self, tk: Token):
        self.var = False
        self.args = []
        super().__init__(tk)

class Assertion(Node):
    type: Type

Modifier = Union[
    Index,
    Slice,
    Selector,
    Arguments,
    Assertion,
]

### Top Level Declarations ###

class InitSpec(Node):
    type     : Optional[Type]
    names    : List[Name]
    values   : List[Expression]
    readonly : bool

    def __init__(self, tk: Token):
        self.type = None
        self.names = []
        self.values = []
        self.readonly = False
        super().__init__(tk)

class TypeSpec(Node):
    name     : Name
    type     : 'Type'
    is_alias : bool

    def __init__(self, tk: Token):
        self.is_alias = False
        super().__init__(tk)

class Function(Node):
    pass    # TODO: define this

class ImportHere(Node):
    def __init__(self, tk: Token):
        super().__init__(tk)
        assert tk.kind == TokenType.Operator and tk.value == '.'

class ImportSpec(Node):
    path  : String
    alias : Optional[Union[Name, ImportHere]]

    def __init__(self, tk: Token):
        self.alias = None
        super().__init__(tk)

class Package(Node):
    name    : Name
    vars    : List[InitSpec]
    funcs   : List[Function]
    types   : List[TypeSpec]
    consts  : List[InitSpec]
    imports : List[ImportSpec]

    def __init__(self, tk: Token):
        self.vars = []
        self.funcs = []
        self.types = []
        self.consts = []
        self.imports = []
        super().__init__(tk)
