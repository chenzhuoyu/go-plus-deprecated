# -*- coding: utf-8 -*-

from enum import IntEnum

from typing import List
from typing import Optional
from typing import FrozenSet

from .utils import StrictFields
from .flags import ChannelOptions
from .flags import FunctionOptions

class Kind(IntEnum):
    Invalid       = 0
    Bool          = 1
    Int           = 2
    Int8          = 3
    Int16         = 4
    Int32         = 5
    Int64         = 6
    Uint          = 7
    Uint8         = 8
    Uint16        = 9
    Uint32        = 10
    Uint64        = 11
    Uintptr       = 12
    Float32       = 13
    Float64       = 14
    Complex64     = 15
    Complex128    = 16
    Array         = 17
    Chan          = 18
    Func          = 19
    Interface     = 20
    Map           = 21
    Ptr           = 22
    Slice         = 23
    String        = 24
    Struct        = 25
    UnsafePointer = 26

class Type(metaclass = StrictFields):
    kind  : Kind
    valid : bool

    __noinit__ = {
        'kind',
        'valid',
    }

    def __init__(self, kind: Kind, valid: bool = True):
        self.kind = kind
        self.valid = valid

    def __repr__(self) -> str:
        return self._to_repr(frozenset())

    def _to_repr(self, _path: FrozenSet['Type']) -> str:
        if self in _path:
            return '...'
        else:
            return self._get_repr(_path | {self})

    def _get_repr(self, _path: FrozenSet['Type']) -> str:
        return '%s@%#x' % (self.kind.name.lower(), id(self))

class PtrType(Type):
    elem: Optional[Type]

    def __init__(self, elem: Optional[Type] = None):
        self.elem = elem
        super().__init__(Kind.Ptr)

    def _get_repr(self, _path: FrozenSet[Type]) -> str:
        return '(*%s)@%#x' % (self.elem._to_repr(_path), id(self))

class MapType(Type):
    key  : Optional[Type]
    elem : Optional[Type]

    def __init__(self, key: Optional[Type] = None, elem: Optional[Type] = None):
        self.key = key
        self.elem = elem
        super().__init__(Kind.Map)

    def _get_repr(self, _path: FrozenSet[Type]) -> str:
        return '(map[%s]%s)@%#x' % (
            self.key._to_repr(_path),
            self.elem._to_repr(_path),
            id(self),
        )

class FuncType(Type):
    var   : bool
    args  : List[Type]
    rets  : List[Type]
    flags : FunctionOptions

    def __init__(self):
        self.flags = FunctionOptions(0)
        super().__init__(Kind.Func)

class ChanType(Type):
    dir  : ChannelOptions
    elem : Optional[Type]

    __chan_type__ = {
        ChannelOptions.BOTH : 'chan',
        ChannelOptions.SEND : 'chan <-',
        ChannelOptions.RECV : '<- chan',
    }

    def __init__(self, elem: Optional[Type] = None):
        self.dir = ChannelOptions.BOTH
        self.elem = elem
        super().__init__(Kind.Chan)

    def _get_repr(self, _path: FrozenSet[Type]) -> str:
        return '(%s %s)@%#x' % (
            self.__chan_type__[self.dir],
            self.elem._to_repr(_path),
            id(self),
        )

class ArrayType(Type):
    len  : Optional[int]
    elem : Optional[Type]

    def __init__(self, elem: Optional[Type] = None):
        self.elem = elem
        super().__init__(Kind.Array, False)

    def _get_repr(self, _path: FrozenSet[Type]) -> str:
        if self.len is None:
            return '([?]%r)@%#x' % (self.elem._to_repr(_path), id(self))
        else:
            return '([%d]%r)@%#x' % (self.len, self.elem._to_repr(_path), id(self))

class SliceType(Type):
    elem: Optional[Type]

    def __init__(self, elem: Optional[Type] = None):
        self.elem = elem
        super().__init__(Kind.Slice)

    def _get_repr(self, _path: FrozenSet[Type]) -> str:
        return '([]%s)@%#x' % (self.elem._to_repr(_path), id(self))

class StructType(Type):
    size   : int
    align  : int
    fields : List['StructField']

    def __init__(self):
        self.size = 0
        self.align = 0
        super().__init__(Kind.Struct, False)

class StructField(metaclass = StrictFields):
    name     : str
    type     : Type
    tags     : Optional[str]
    size     : int
    align    : int
    offset   : int
    embedded : bool

class InterfaceType(Type):
    methods: List['InterfaceMethod']

    def __init__(self):
        super().__init__(Kind.Interface)

class InterfaceMethod(metaclass = StrictFields):
    name: str
    type: FuncType

    def __init__(self, name: str, mtype: FuncType):
        self.name = name
        self.type = mtype

class NamedType(Type):
    name: str
    type: Optional[Type]

    def __init__(self, name: str, rtype: Optional[Type] = None):
        self.name = name
        self.type = rtype
        super().__init__(Kind.Invalid, False)

    def _get_repr(self, _path: FrozenSet[Type]) -> str:
        return 'named<%s: %r>@%#x' % (self.name, self.type._to_repr(_path), id(self))

class UntypedType(Type):
    def _get_repr(self, _path: FrozenSet[Type]) -> str:
        return 'untyped<%s>' % super()._to_repr(_path)

class Types:
    Bool           = Type(Kind.Bool)
    Int            = Type(Kind.Int)
    Int8           = Type(Kind.Int8)
    Int16          = Type(Kind.Int16)
    Int32          = Type(Kind.Int32)
    Int64          = Type(Kind.Int64)
    Uint           = Type(Kind.Uint)
    Uint8          = Type(Kind.Uint8)
    Uint16         = Type(Kind.Uint16)
    Uint32         = Type(Kind.Uint32)
    Uint64         = Type(Kind.Uint64)
    Uintptr        = Type(Kind.Uintptr)
    Float32        = Type(Kind.Float32)
    Float64        = Type(Kind.Float64)
    Complex64      = Type(Kind.Complex64)
    Complex128     = Type(Kind.Complex128)
    String         = Type(Kind.String)
    UnsafePointer  = Type(Kind.UnsafePointer)

    ### Untyped Types for Constants ###

    UntypedInt     = UntypedType(Kind.Int)
    UntypedBool    = UntypedType(Kind.Bool)
    UntypedRune    = UntypedType(Kind.Int32)
    UntypedFloat   = UntypedType(Kind.Float64)
    UntypedString  = UntypedType(Kind.String)
    UntypedComplex = UntypedType(Kind.Complex128)
