# -*- coding: utf-8 -*-

import json
import inspect

from enum import IntFlag

from typing import Any
from typing import Set
from typing import Dict
from typing import List
from typing import Union
from typing import Optional
from typing import Sequence

from .tokenizer import Token
from .tokenizer import TokenType

class Node:
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
        ret = {}
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

class Name(Node):
    name: str

class String(Node):
    value: bytes

    def __init__(self, tk: Token):
        super().__init__(tk)
        self.value = tk.value
        assert tk.kind == TokenType.String

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
    pass

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
    pass

class InterfaceType(Node):
    pass

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

class Expression(Node):
    pass

### Top Level Declarations ###

class VarSpec(Node):
    pass

class TypeSpec(Node):
    pass

class Function(Node):
    pass

class ConstSpec(Node):
    name  : Name
    type  : Optional[Type]
    value : Expression

    def __init__(self, tk: Token):
        self.type = None
        super().__init__(tk)

class ImportSpec(Node):
    path  : String
    alias : Optional[Name]

    def __init__(self, tk: Token):
        self.alias = None
        super().__init__(tk)

class Package(Node):
    name    : Name
    vars    : Dict[str, VarSpec]
    funcs   : Dict[str, Function]
    types   : Dict[str, TypeSpec]
    consts  : Dict[str, ConstSpec]
    imports : List[ImportSpec]

    def __init__(self, tk: Token):
        self.vars = {}
        self.funcs = {}
        self.types = {}
        self.consts = {}
        self.imports = []
        super().__init__(tk)
