# -*- coding: utf-8 -*-

from enum import IntEnum

from typing import cast
from typing import List
from typing import Optional
from typing import FrozenSet

from .utils import StrictFields
from .flags import ChannelOptions
from .flags import FunctionOptions

class Kind(IntEnum):
    Nil           = 0
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
    Named         = 255

class Type(metaclass = StrictFields):
    kind   : Kind
    valid  : bool
    tfuncs : List['Method']
    pfuncs : List['Method']

    __noinit__ = {
        'kind',
        'valid',
    }

    def __init__(self, kind: Kind, valid: bool = True):
        self.kind = kind
        self.valid = valid

    def __repr__(self) -> str:
        return self._to_repr(frozenset())

    def __hash__(self) -> int:
        return self.kind.value

    def __ne__(self, other: 'Type') -> bool:
        return not (self == other)

    def __eq__(self, other: 'Type') -> bool:
        return isinstance(other, Type) and \
               self.valid and \
               other.valid and \
               self.kind == other.kind and \
               self.tfuncs == other.tfuncs and \
               self.pfuncs == other.pfuncs

    def _to_repr(self, _path: FrozenSet['Type']) -> str:
        if id(self) in _path:
            return '...'
        else:
            return self._get_repr(_path | {id(self)})

    def _get_repr(self, _path: FrozenSet['Type']) -> str:
        return self.kind.name.lower()

class Method(metaclass = StrictFields):
    name: str
    type: 'FuncType'

    def __init__(self, name: str, mtype: 'FuncType'):
        self.name = name
        self.type = mtype

    def __eq__(self, val: 'Method') -> bool:
        return self.name == val.name and self.type == val.type

class PtrType(Type):
    elem: Optional[Type]

    def __init__(self, elem: Optional[Type] = None):
        self.elem = elem
        super().__init__(Kind.Ptr)

    def __hash__(self) -> int:
        return hash(self.elem)

    def __eq__(self, val: 'Type') -> bool:
        return super().__eq__(val) and \
               isinstance(val, PtrType) and \
               self.elem == cast(PtrType, val).elem

    def _get_repr(self, _path: FrozenSet[Type]) -> str:
        return '*%s' % self.elem._to_repr(_path)

class MapType(Type):
    key  : Optional[Type]
    elem : Optional[Type]

    def __init__(self, key: Optional[Type] = None, elem: Optional[Type] = None):
        self.key = key
        self.elem = elem
        super().__init__(Kind.Map)

    def __hash__(self) -> int:
        return hash((self.key, self.elem))

    def __eq__(self, val: 'Type') -> bool:
        return super().__eq__(val) and \
               isinstance(val, MapType) and \
               self.key == cast(MapType, val).key and \
               self.elem == cast(MapType, val).elem

    def _get_repr(self, _path: FrozenSet[Type]) -> str:
        return 'map[%s]%s' % (
            self.key._to_repr(_path),
            self.elem._to_repr(_path),
        )

class FuncType(Type):
    var   : bool
    args  : List[Type]
    rets  : List[Type]
    flags : FunctionOptions

    def __init__(self):
        self.flags = FunctionOptions(0)
        super().__init__(Kind.Func)

    def __hash__(self) -> int:
        return hash((self.var, self.flags))

    def __eq__(self, val: 'Type') -> bool:
        return super().__eq__(val) and \
               isinstance(val, FuncType) and \
               self.var == cast(FuncType, val).var and \
               self.args == cast(FuncType, val).args and \
               self.rets == cast(FuncType, val).rets and \
               self.flags == cast(FuncType, val).flags

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

    def __hash__(self) -> int:
        return hash((self.dir, self.elem))

    def __eq__(self, val: 'Type') -> bool:
        return super().__eq__(val) and \
               isinstance(val, ChanType) and \
               self.dir == cast(ChanType, val).dir and \
               self.elem == cast(ChanType, val).elem

    def _get_repr(self, _path: FrozenSet[Type]) -> str:
        return '%s %s' % (
            self.__chan_type__[self.dir],
            self.elem._to_repr(_path),
        )

class ArrayType(Type):
    len  : Optional[int]
    elem : Optional[Type]

    def __init__(self, elem: Optional[Type] = None):
        self.elem = elem
        super().__init__(Kind.Array, False)

    def __hash__(self) -> int:
        return hash((self.len, self.elem))

    def __eq__(self, val: 'Type') -> bool:
        return super().__eq__(val) and \
               isinstance(val, ArrayType) and \
               self.len == cast(ArrayType, val).len and \
               self.elem == cast(ArrayType, val).elem

    def _get_repr(self, _path: FrozenSet[Type]) -> str:
        if self.len is None:
            return '[?]%s' % self.elem._to_repr(_path)
        else:
            return '[%d]%s' % (self.len, self.elem._to_repr(_path))

class SliceType(Type):
    elem: Optional[Type]

    def __init__(self, elem: Optional[Type] = None):
        self.elem = elem
        super().__init__(Kind.Slice)

    def __hash__(self) -> int:
        return hash(self.elem)

    def __eq__(self, val: 'Type') -> bool:
        return super().__eq__(val) and \
               isinstance(val, SliceType) and \
               self.elem == cast(SliceType, val).elem

    def _get_repr(self, _path: FrozenSet[Type]) -> str:
        return '[]%s' % self.elem._to_repr(_path)

class StructType(Type):
    fields: List['StructField']

    def __init__(self):
        super().__init__(Kind.Struct, False)

    def __eq__(self, val: 'Type') -> bool:
        return super().__eq__(val) and \
               isinstance(val, StructType) and \
               self.fields == cast(StructType, val).fields

class StructField(metaclass = StrictFields):
    name  : str
    type  : Type
    tags  : Optional[str]
    embed : bool

    def __eq__(self, val: 'StructField') -> bool:
        return self.name == val.name and \
               self.type == val.type and \
               self.tags == val.tags and \
               self.embed == val.embed

class InterfaceType(Type):
    def __init__(self):
        super().__init__(Kind.Interface)

class NamedType(Type):
    name: str
    type: Optional[Type]

    def __init__(self, name: str, rtype: Optional[Type] = None):
        self.name = name
        self.type = rtype
        super().__init__(Kind.Named, False)

    def __hash__(self) -> int:
        return hash(self.type)

    def __eq__(self, val: 'Type') -> bool:
        if not isinstance(val, NamedType):
            return self.type is val
        else:
            return self.type is val.type

    @property
    def kind(self) -> Kind:
        if self.type is None:
            return Kind.Named
        else:
            return self.type.kind

    @property
    def valid(self) -> bool:
        if self.type is None:
            return False
        else:
            return self.type.valid

    kind  = kind.setter(lambda *_: None)
    valid = valid.setter(lambda *_: None)

    def _get_repr(self, _path: FrozenSet[Type]) -> str:
        return '%s(%s)' % (self.name, self.type._get_repr(_path))

class UntypedType(Type):
    def __eq__(self, other: 'Type') -> bool:
        return super().__eq__(other) and isinstance(other, UntypedType)

    def __hash__(self) -> int:
        return super().__hash__()

    def _get_repr(self, _path: FrozenSet[Type]) -> str:
        return 'untyped %s' % super()._get_repr(_path)

class Types:
    Nil            = Type(Kind.Nil)
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
    UntypedFloat   = UntypedType(Kind.Float64)
    UntypedString  = UntypedType(Kind.String)
    UntypedComplex = UntypedType(Kind.Complex128)
