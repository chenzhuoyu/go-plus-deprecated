# -*- coding: utf-8 -*-

from enum import IntEnum

from typing import Type as Tp
from typing import List
from typing import Optional

from .utils import StrictFields
from .flags import ChannelOptions
from .flags import FunctionOptions

class Kind(IntEnum):
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
    kind: Kind

    ### Standard Types ###

    Bool           : 'Type'
    Int            : 'Type'
    Int8           : 'Type'
    Int16          : 'Type'
    Int32          : 'Type'
    Int64          : 'Type'
    Uint           : 'Type'
    Uint8          : 'Type'
    Uint16         : 'Type'
    Uint32         : 'Type'
    Uint64         : 'Type'
    Uintptr        : 'Type'
    Float32        : 'Type'
    Float64        : 'Type'
    Complex64      : 'Type'
    Complex128     : 'Type'
    Array          : Tp['ArrayType']
    Chan           : Tp['ChanType']
    Func           : Tp['FuncType']
    Interface      : Tp['InterfaceType']
    Map            : Tp['MapType']
    Ptr            : Tp['PtrType']
    Slice          : Tp['SliceType']
    String         : 'Type'
    Struct         : Tp['StructType']
    UnsafePointer  : 'Type'

    ### Untyped Types for Constants ###

    UntypedInt     : 'UntypedType'
    UntypedBool    : 'UntypedType'
    UntypedRune    : 'UntypedType'
    UntypedFloat   : 'UntypedType'
    UntypedString  : 'UntypedType'
    UntypedComplex : 'UntypedType'

    def __init__(self, kind: Kind):
        self.kind = kind

    def __str__(self) -> str:
        return self.kind.name.lower()

class PtrType(Type):
    elem: Type

    def __init__(self, elem: Type):
        self.elem = elem
        super().__init__(Kind.Ptr)

    def __str__(self) -> str:
        return '*' + str(self.elem)

class MapType(Type):
    key  : Type
    elem : Type

    def __init__(self, key: Type, elem: Type):
        self.key = key
        self.elem = elem
        super().__init__(Kind.Map)

    def __str__(self) -> str:
        return 'map[%s]%s' % (self.key, self.elem)

class FuncType(Type):
    var   : bool
    args  : List[Type]
    rets  : List[Type]
    flags : FunctionOptions

    def __init__(self):
        self.var = False
        self.flags = FunctionOptions(0)
        super().__init__(Kind.Func)

class ChanType(Type):
    dir  : ChannelOptions
    elem : Type

    def __init__(self, elem: Type):
        self.dir = ChannelOptions.BOTH
        self.elem = elem
        super().__init__(Kind.Chan)

    def __str__(self) -> str:
        if self.dir == ChannelOptions.BOTH:
            return 'chan %s' % self.elem
        elif self.dir == ChannelOptions.SEND:
            return 'chan <- %s' % self.elem
        else:
            return '<- chan %s' % self.elem

class ArrayType(Type):
    len  : Optional[int]
    elem : Type

    def __init__(self, elem: Type):
        self.elem = elem
        super().__init__(Kind.Array)

    def __str__(self) -> str:
        if self.len is None:
            return '[...]%s' % self.elem
        else:
            return '[%d]%s' % (self.len, self.elem)

class SliceType(Type):
    elem: Type

    def __init__(self, elem: Type):
        self.elem = elem
        super().__init__(Kind.Slice)

    def __str__(self) -> str:
        return '[]%s' % self.elem

class StructType(Type):
    fields: List['StructField']

    def __init__(self):
        super().__init__(Kind.Struct)

class StructField(metaclass = StrictFields):
    name: str
    anon: bool
    type: Type
    tags: Optional[str]

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

class UntypedType(Type):
    def __str__(self) -> str:
        return 'untyped<%s>' % super().__str__()

class InheritedType(Type):
    type: Type

    def __init__(self, rtype: Type):
        self.type = rtype
        super().__init__(rtype.kind)

    def __str__(self) -> str:
        return 'inherited<%s>' % str(self.type)

Type.Bool           = Type(Kind.Bool)
Type.Int            = Type(Kind.Int)
Type.Int8           = Type(Kind.Int8)
Type.Int16          = Type(Kind.Int16)
Type.Int32          = Type(Kind.Int32)
Type.Int64          = Type(Kind.Int64)
Type.Uint           = Type(Kind.Uint)
Type.Uint8          = Type(Kind.Uint8)
Type.Uint16         = Type(Kind.Uint16)
Type.Uint32         = Type(Kind.Uint32)
Type.Uint64         = Type(Kind.Uint64)
Type.Uintptr        = Type(Kind.Uintptr)
Type.Float32        = Type(Kind.Float32)
Type.Float64        = Type(Kind.Float64)
Type.Complex64      = Type(Kind.Complex64)
Type.Complex128     = Type(Kind.Complex128)
Type.Array          = ArrayType
Type.Chan           = ChanType
Type.Func           = FuncType
Type.Interface      = InterfaceType
Type.Map            = MapType
Type.Ptr            = PtrType
Type.Slice          = SliceType
Type.String         = Type(Kind.String)
Type.Struct         = StructType
Type.UnsafePointer  = Type(Kind.UnsafePointer)

Type.UntypedInt     = UntypedType(Kind.Int)
Type.UntypedBool    = UntypedType(Kind.Bool)
Type.UntypedRune    = UntypedType(Kind.Int32)
Type.UntypedFloat   = UntypedType(Kind.Float64)
Type.UntypedString  = UntypedType(Kind.String)
Type.UntypedComplex = UntypedType(Kind.Complex128)
