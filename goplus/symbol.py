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

from .types import Method
from .types import FuncType
from .types import InterfaceType

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

def _make_intf(methods: List[Method]) -> InterfaceType:
    ret = InterfaceType()
    ret.tfuncs = methods
    return ret

def _make_func(args: List[Type], rets: List[Type]) -> FuncType:
    ret = FuncType()
    ret.args = args
    ret.rets = rets
    ret.flags = 0
    return ret

class Functions:
    Append  = Symbols.Func('append'  , FuncType())
    Copy    = Symbols.Func('copy'    , FuncType())
    Delete  = Symbols.Func('delete'  , FuncType())
    Len     = Symbols.Func('len'     , FuncType())
    Cap     = Symbols.Func('cap'     , FuncType())
    Make    = Symbols.Func('make'    , FuncType())
    New     = Symbols.Func('new'     , FuncType())
    Complex = Symbols.Func('complex' , FuncType())
    Real    = Symbols.Func('real'    , FuncType())
    Imag    = Symbols.Func('imag'    , FuncType())
    Close   = Symbols.Func('close'   , FuncType())
    Panic   = Symbols.Func('panic'   , FuncType())
    Recover = Symbols.Func('recover' , FuncType())
    Print   = Symbols.Func('print'   , FuncType())
    PrintLn = Symbols.Func('println' , FuncType())

class Interfaces:
    Error = _make_intf([Method(
        'Error',
        _make_func([], [Types.String])
    )])

class ConstValue(Symbols.Const):
    value: TokenValue

    def __init__(self, name: str, vtype: Type, value: TokenValue):
        self.value = value
        super().__init__(name, vtype)

BUILTIN_SYMBOLS = {
    'nil'        : ConstValue('nil'   , Types.Nil, None),
    'true'       : ConstValue('true'  , Types.UntypedBool, True),
    'false'      : ConstValue('false' , Types.UntypedBool, False),

    'append'     : Functions.Append,
    'copy'       : Functions.Copy,
    'delete'     : Functions.Delete,
    'len'        : Functions.Len,
    'cap'        : Functions.Cap,
    'make'       : Functions.Make,
    'new'        : Functions.New,
    'complex'    : Functions.Complex,
    'real'       : Functions.Real,
    'imag'       : Functions.Imag,
    'close'      : Functions.Close,
    'panic'      : Functions.Panic,
    'recover'    : Functions.Recover,
    'print'      : Functions.Print,
    'println'    : Functions.PrintLn,

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

    'byte'       : Symbols.Type('byte'  , Types.Uint8),
    'rune'       : Symbols.Type('rune'  , Types.Int32),
    'error'      : Symbols.Type('error' , Interfaces.Error),
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
