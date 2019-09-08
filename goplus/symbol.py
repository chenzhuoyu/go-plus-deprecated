# -*- coding: utf-8 -*-

from enum import Enum

from typing import Dict
from typing import List
from typing import Optional

from .ast import Package
from .ast import TokenValue

from .types import Type
from .types import Types
from .utils import StrictFields

class SymbolKind(Enum):
    VAR     = 'var'
    FUNC    = 'func'
    TYPE    = 'type'
    CONST   = 'const'
    PACKAGE = 'package'

class Symbol(metaclass = StrictFields):
    name: str
    kind: SymbolKind
    type: Optional[Type]

    def __init__(self, name: str, rtype: Optional[Type]):
        self.name = name
        self.type = rtype

    def __repr__(self) -> str:
        if self.type is None:
            return '#{%s: %s}' % (self.name, self.kind.value)
        else:
            return '#{%s: %s<%s>}' % (self.name, self.kind.value, str(self.type))

class Symbols:
    class Var(Symbol):     kind = SymbolKind.VAR
    class Func(Symbol):    kind = SymbolKind.FUNC
    class Type(Symbol):    kind = SymbolKind.TYPE
    class Const(Symbol):   kind = SymbolKind.CONST
    class Package(Symbol): kind = SymbolKind.PACKAGE

class ConstValue(Symbols.Const):
    value: TokenValue

    def __init__(self, name: str, vtype: Type, value: TokenValue):
        self.value = value
        super().__init__(name, vtype)

BUILTIN_SYMBOLS = {
    'true'       : ConstValue('true'  , Types.UntypedBool, True),
    'false'      : ConstValue('false' , Types.UntypedBool, False),

    'bool'       : Symbols.Type('bool'       , Types.Bool),
    'int'        : Symbols.Type('int'        , Types.Int),
    'int8'       : Symbols.Type('int8'       , Types.Int8),
    'int16'      : Symbols.Type('int16'      , Types.Int16),
    'int32'      : Symbols.Type('int32'      , Types.Int32),
    'int64'      : Symbols.Type('int64'      , Types.Int64),
    'uint'       : Symbols.Type('uint'       , Types.Uint),
    'uint8'      : Symbols.Type('uint8'      , Types.Uint8),
    'uint16'     : Symbols.Type('uint16'     , Types.Uint16),
    'uint32'     : Symbols.Type('uint32'     , Types.Uint32),
    'uint64'     : Symbols.Type('uint64'     , Types.Uint64),
    'uintptr'    : Symbols.Type('uintptr'    , Types.Uintptr),
    'float32'    : Symbols.Type('float32'    , Types.Float32),
    'float64'    : Symbols.Type('float64'    , Types.Float64),
    'complex64'  : Symbols.Type('complex64'  , Types.Complex64),
    'complex128' : Symbols.Type('complex128' , Types.Complex128),
    'string'     : Symbols.Type('string'     , Types.String),
}

class Scope(metaclass = StrictFields):
    def resolve(self, name: str) -> Optional[Symbol]:
        raise NotImplementedError

    def declare(self, name: str, sym: Symbol) -> bool:
        raise NotImplementedError

class BlockScope(Scope):
    parent  : Scope
    symbols : Dict[str, Symbol]

    def __init__(self, parent: Scope):
        self.parent = parent
        self.symbols = {}

    def derive(self) -> 'BlockScope':
        return BlockScope(self)

    def resolve(self, name: str) -> Optional[Symbol]:
        if name in self.symbols:
            return self.symbols[name]
        else:
            return self.parent.resolve(name)

    def declare(self, name: str, sym: Symbol) -> bool:
        if name in self.symbols:
            return False
        else:
            self.symbols[name] = sym
            return True

class GlobalScope(Scope):
    def resolve(self, name: str) -> Optional[Symbol]:
        return BUILTIN_SYMBOLS.get(name, None)

    def declare(self, name: str, sym: Symbol) -> bool:
        raise SystemError('cannot declare in global scope')

class PackageScope(Scope, Symbols.Package):
    path    : str                       # package path, like "example.com/example/pkg"
    files   : List[Package]             # parsed AST roots
    parent  : GlobalScope               # parent scope (must be `GlobalScope`)
    public  : Dict[str, Symbol]         # exported symbols of this package
    shared  : Dict[str, Symbol]         # symbols that are not exported, but visible across all files
    private : Dict[str, BlockScope]     # symbols that are private to the current file

    def __init__(self, name: str, path: str):
        self.path = path
        self.parent = GlobalScope()
        super().__init__(name, None)

    def source(self, name: str) -> Scope:
        if name in self.private:
            return self.private[name]
        else:
            return self.private.setdefault(name, BlockScope(self))

    def resolve(self, name: str) -> Optional[Symbol]:
        if name in self.public:
            return self.public[name]
        elif name in self.shared:
            return self.shared[name]
        else:
            return self.parent.resolve(name)

    def declare(self, name: str, sym: Symbol) -> bool:
        if name in self.public:
            return False
        elif name in self.shared:
            return False
        elif 'A' <= name[0] <= 'Z':
            self.public[name] = sym
            return True
        else:
            self.shared[name] = sym
            return True
