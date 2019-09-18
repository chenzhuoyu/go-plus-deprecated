# -*- coding: utf-8 -*-

import os
import enum
import math
import operator
import functools

from typing import Set
from typing import Dict
from typing import List
from typing import Tuple
from typing import Union
from typing import TextIO
from typing import Callable
from typing import Iterable
from typing import Optional

from .ast import Int
from .ast import Bool
from .ast import Name
from .ast import Node
from .ast import Rune
from .ast import Float
from .ast import String
from .ast import Complex
from .ast import Package
from .ast import Operator

from .ast import Primary
from .ast import Constant
from .ast import Selector
from .ast import Arguments
from .ast import Conversion
from .ast import Expression

from .ast import InitSpec
from .ast import TypeSpec
from .ast import Function

from .ast import ImportC
from .ast import ImportHere

from .ast import Type as TypeNode
from .ast import MapType as MapTypeNode
from .ast import ArrayType as ArrayTypeNode
from .ast import SliceType as SliceTypeNode
from .ast import NamedType as NamedTypeNode
from .ast import StructType as StructTypeNode
from .ast import ChannelType as ChannelTypeNode
from .ast import PointerType as PointerTypeNode
from .ast import VarArrayType as VarArrayTypeNode
from .ast import FunctionType as FunctionTypeNode
from .ast import InterfaceType as InterfaceTypeNode

from .types import Kind
from .types import Type
from .types import Types

from .types import MapType
from .types import PtrType
from .types import ChanType
from .types import FuncType
from .types import ArrayType
from .types import SliceType
from .types import NamedType
from .types import StructType
from .types import UntypedType
from .types import InterfaceType

from .symbol import Scope
from .symbol import Symbol
from .symbol import Symbols
from .symbol import Functions
from .symbol import ConstValue
from .symbol import PackageScope

from .modules import Module
from .modules import Reader
from .modules import Resolver

from .tokenizer import Token
from .tokenizer import TokenType
from .tokenizer import TokenValue

from .parser import Parser
from .tokenizer import Tokenizer

GOOS = {
    'aix',
    'android',
    'darwin',
    'dragonfly',
    'freebsd',
    'hurd',
    'illumos',
    'js',
    'linux',
    'nacl',
    'netbsd',
    'openbsd',
    'plan9',
    'solaris',
    'windows',
    'zos',
}

GOARCH = {
    '386',
    'amd64',
    'amd64p32',
    'arm',
    'armbe',
    'arm64',
    'arm64be',
    'ppc64',
    'ppc64le',
    'mips',
    'mipsle',
    'mips64',
    'mips64le',
    'mips64p32',
    'mips64p32le',
    'ppc',
    'riscv',
    'riscv64',
    's390',
    's390x',
    'sparc',
    'sparc64',
    'wasm',
}

GO_VERS = [
    'go1.1',
    'go1.2',
    'go1.3',
    'go1.4',
    'go1.5',
    'go1.6',
    'go1.7',
    'go1.8',
    'go1.9',
    'go1.10',
    'go1.11',
    'go1.12',
    'go1.13',
]

GO_EXTRA = {
    'android': ['linux'],
    'illumos': ['solaris'],
}

CGO_ENABLED = {
    'aix/ppc64',
    'android/386',
    'android/amd64',
    'android/arm',
    'android/arm64',
    'darwin/amd64',
    'darwin/arm',
    'darwin/arm64',
    'dragonfly/amd64',
    'freebsd/386',
    'freebsd/amd64',
    'freebsd/arm',
    'illumos/amd64',
    'linux/386',
    'linux/amd64',
    'linux/arm',
    'linux/arm64',
    'linux/mips',
    'linux/mips64',
    'linux/mips64le',
    'linux/mipsle',
    'linux/ppc64le',
    'linux/riscv64',
    'linux/s390x',
    'linux/sparc64',
    'netbsd/386',
    'netbsd/amd64',
    'netbsd/arm',
    'netbsd/arm64',
    'openbsd/386',
    'openbsd/amd64',
    'openbsd/arm',
    'openbsd/arm64',
    'solaris/amd64',
    'windows/386',
    'windows/amd64',
}

INT_KINDS = {
    Kind.Int     : TokenType.Int,
    Kind.Int8    : TokenType.Int,
    Kind.Int16   : TokenType.Int,
    Kind.Int32   : TokenType.Int,
    Kind.Int64   : TokenType.Int,
    Kind.Uint    : TokenType.Int,
    Kind.Uint8   : TokenType.Int,
    Kind.Uint16  : TokenType.Int,
    Kind.Uint32  : TokenType.Int,
    Kind.Uint64  : TokenType.Int,
    Kind.Uintptr : TokenType.Int,
}

FLOAT_KINDS = {
    Kind.Float32: TokenType.Float,
    Kind.Float64: TokenType.Float,
}

STRING_KINDS = {
    Kind.String: TokenType.String,
}

COMPLEX_KINDS = {
    Kind.Complex64  : TokenType.Complex,
    Kind.Complex128 : TokenType.Complex,
}

BOOLEAN_KINDS = {
    Kind.Bool: TokenType.Bool,
}

NUMERIC_KINDS = {
    **INT_KINDS,
    **FLOAT_KINDS,
    **COMPLEX_KINDS,
}

GENERIC_KINDS = {
    **STRING_KINDS,
    **NUMERIC_KINDS,
}

COERCING_MAPS = {
    Types.UntypedInt: {
        Kind.Int,
        Kind.Int8,
        Kind.Int16,
        Kind.Int32,
        Kind.Int64,
        Kind.Uint,
        Kind.Uint8,
        Kind.Uint16,
        Kind.Uint32,
        Kind.Uint64,
        Kind.Float32,
        Kind.Float64,
        Kind.Complex64,
        Kind.Complex128,
    },
    Types.UntypedBool: {
        Kind.Bool,
    },
    Types.UntypedFloat: {
        Kind.Float32,
        Kind.Float64,
        Kind.Complex64,
        Kind.Complex128,
    },
    Types.UntypedString: {
        Kind.String,
    },
    Types.UntypedComplex: {
        Kind.Complex64,
        Kind.Complex128,
    },
}

def _is_f32(v: float) -> bool:
    return -3.402823466e+38 <= v <= 3.402823466e+38

def _is_f64(v: float) -> bool:
    return -1.7976931348623158e+308 <= v <= 1.7976931348623158e+308

LITERAL_RANGES = {
    Kind.Int        : lambda v: -0x8000000000000000 <= v <= 0x7fffffffffffffff,
    Kind.Int8       : lambda v: -0x80               <= v <= 0x7f,
    Kind.Int16      : lambda v: -0x8000             <= v <= 0x7fff,
    Kind.Int32      : lambda v: -0x80000000         <= v <= 0x7fffffff,
    Kind.Int64      : lambda v: -0x8000000000000000 <= v <= 0x7fffffffffffffff,
    Kind.Uint       : lambda v: 0 <= v <= 0xffffffffffffffff,
    Kind.Uint8      : lambda v: 0 <= v <= 0xff,
    Kind.Uint16     : lambda v: 0 <= v <= 0xffff,
    Kind.Uint32     : lambda v: 0 <= v <= 0xffffffff,
    Kind.Uint64     : lambda v: 0 <= v <= 0xffffffffffffffff,
    Kind.Uintptr    : lambda v: 0 <= v <= 0xffffffffffffffff,
    Kind.Float32    : _is_f32,
    Kind.Float64    : _is_f64,
    Kind.Complex64  : lambda v: _is_f32(v.real) and _is_f32(v.imag),
    Kind.Complex128 : lambda v: _is_f64(v.real) and _is_f64(v.imag),
    Kind.Bool       : lambda v: False,
    Kind.String     : lambda _: True,
}

CONVERTING_MAPS = {
    Kind.Int        : int,
    Kind.Int8       : int,
    Kind.Int16      : int,
    Kind.Int32      : int,
    Kind.Int64      : int,
    Kind.Uint       : int,
    Kind.Uint8      : int,
    Kind.Uint16     : int,
    Kind.Uint32     : int,
    Kind.Uint64     : int,
    Kind.Uintptr    : int,
    Kind.Float32    : float,
    Kind.Float64    : float,
    Kind.Complex64  : complex,
    Kind.Complex128 : complex,
    Kind.Bool       : bool,
    Kind.String     : bytes,
}

CONSTRUCTING_MAPS = {
    Kind.Int        : Int,
    Kind.Int8       : Int,
    Kind.Int16      : Int,
    Kind.Int32      : Int,
    Kind.Int64      : Int,
    Kind.Uint       : Int,
    Kind.Uint8      : Int,
    Kind.Uint16     : Int,
    Kind.Uint32     : Int,
    Kind.Uint64     : Int,
    Kind.Uintptr    : Int,
    Kind.Float32    : Float,
    Kind.Float64    : Float,
    Kind.Complex64  : Complex,
    Kind.Complex128 : Complex,
    Kind.Bool       : Bool,
    Kind.String     : String,
}

PackageMap = Dict[
    str,
    Package,
]

NumericType = Union[
    int,
    float,
    complex,
]

class Mode(enum.IntEnum):
    GO_MOD    = 0
    GO_VENDOR = 1

class Backend(enum.Enum):
    GC    = 'gc'
    GCCGO = 'gccgo'

class Combinator(enum.Enum):
    OR  = False
    AND = True

class Ops:
    @staticmethod
    def div(a: NumericType, b: NumericType) -> NumericType:
        if isinstance(a, (float, complex)):
            return a / b
        elif isinstance(b, (float, complex)):
            return a / b
        else:
            return a // b

    @staticmethod
    def and_not(a: int, b: int) -> int:
        return a & ~b

    @staticmethod
    def bool_or(a: bool, b: bool) -> bool:
        return a or b

    @staticmethod
    def bool_and(a: bool, b: bool) -> bool:
        return a and b

class Tag:
    name   : str
    invert : bool

    def __init__(self, name: str, invert: bool):
        self.name = name
        self.invert = invert

    def __repr__(self) -> str:
        if not self.invert:
            return self.name
        else:
            return '(NOT %s)' % self.name

    def eval(self, tagv: Set[str]) -> bool:
        return self.invert is (self.name not in tagv)

class Tags:
    comb: bool
    tags: List[Union[Tag, 'Tags']]

    def __init__(self, comb: Combinator):
        self.tags = []
        self.comb = comb.value

    def __repr__(self) -> str:
        if not self.tags:
            return '(TRUE)'
        elif len(self.tags) == 1:
            return repr(self.tags[0])
        elif not self.comb:
            return '(%s)' % ' OR '.join(map(repr, self.tags))
        else:
            return '(%s)' % ' AND '.join(map(repr, self.tags))

    def eval(self, tagv: Set[str]) -> bool:
        ret = self.comb
        tags = self.tags

        # combine every tag
        while tags and ret == self.comb:
            ret = tags[0].eval(tagv)
            tags = tags[1:]

        # all done
        return ret

class Trace:
    path  : str
    trace : List[str]

    def __init__(self, trace: List[str], path: str):
        self.path = path
        self.trace = trace

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.trace.remove(self.path)

    def __enter__(self) -> 'Trace':
        self.trace.append(self.path)
        return self

class NameGen:
    index  : int
    prefix : str

    @classmethod
    def next(cls) -> str:
        cls.index += 1
        return cls.prefix + str(cls.index)

    @classmethod
    def verify(cls, name: str) -> bool:
        return name.startswith(cls.prefix)

class InitGen(NameGen):
    index = -1
    prefix = '$_init_'

class BlankGen(NameGen):
    index = -1
    prefix = '$_blank_'

class InplaceGen(NameGen):
    index = -1
    prefix = '$_inplace_'

class Inferrer:
    os      : str
    arch    : str
    proj    : str
    root    : str
    test    : bool
    mode    : Mode
    iota    : Optional[int]
    tags    : Set[str]
    paths   : List[str]
    backend : Backend

    class Iota:
        ifr  : 'Inferrer'
        iota : Optional[int]

        def __init__(self, ifr: 'Inferrer', iota: Optional[int]):
            self.ifr = ifr
            self.iota = iota

        def __exit__(self, exc_type, exc_val, exc_tb):
            self.ifr.iota = self.iota

        def __enter__(self):
            self.ifr.iota, self.iota = self.iota, self.ifr.iota
            return self

    class Context:
        pkg   : PackageScope
        fmap  : PackageMap
        scope : Scope

        def __init__(self, pkg: PackageScope, fmap: PackageMap, file: Package):
            self.pkg = pkg
            self.fmap = fmap
            self.scope = pkg.source(file.file)

    def __init__(self, osn: str, arch: str, proj: str, root: str, paths: List[str]):
        self.os      = osn
        self.arch    = arch
        self.proj    = proj
        self.root    = root
        self.test    = False
        self.mode    = Mode.GO_MOD
        self.iota    = None
        self.tags    = set()
        self.paths   = paths
        self.backend = Backend.GC

    ### Helper Functions ###

    def _error(self, node: Node, msg: str) -> SyntaxError:
        return SyntaxError('%s:%d:%d: %s' % (node.file, node.row + 1, node.col + 1, msg))

    def _string(self, val: str) -> String:
        return String(Token(
            0,
            0,
            '<main>',
            TokenType.String,
            val.encode('utf-8'),
        ))

    ### Symbol Mapping ###

    def _map_name(self, pkg: Package, fmap: PackageMap, name: str, node: Name):
        if name != '_':
            if name not in fmap:
                fmap[name] = pkg
            else:
                raise self._error(node, '%s redeclared in this package' % repr(name))

    def _map_spec_cv(self, pkg: Package, fmap: PackageMap, spec: List[InitSpec]):
        for item in spec:
            for name in item.names:
                if name.value != 'init':
                    self._map_name(pkg, fmap, name.value, name)
                else:
                    raise self._error(name, 'cannot declare init - must be func')

    def _map_spec_tp(self, pkg: Package, fmap: PackageMap, spec: List[TypeSpec]):
        for item in spec:
            if item.name.value != 'init':
                self._map_name(pkg, fmap, item.name.value, item.name)
            else:
                raise self._error(item.name, 'cannot declare init - must be func')

    def _map_spec_fn(self, pkg: Package, fmap: PackageMap, spec: List[Function]):
        for item in spec:
            if item.recv is not None:
                self._map_name(pkg, fmap, self._make_method_name(item), item.name)
            elif item.name.value != 'init':
                self._map_name(pkg, fmap, item.name.value, item.name)

    def _make_type_name(self, vtype: TypeNode) -> str:
        if isinstance(vtype, NamedTypeNode):
            return vtype.name.value
        elif isinstance(vtype, PointerTypeNode) and isinstance(vtype.base, NamedTypeNode):
            return vtype.base.name.value
        else:
            raise self._error(vtype, 'unnamed receiver type')

    def _make_method_name(self, method: Function) -> str:
        rt = method.recv.type
        fn = method.name.value
        tn = self._make_type_name(rt)

        # special case for blank idenfitifers
        if fn == '_':
            return '_'
        else:
            return '%s.%s' % (tn, fn)

    ### Symbol Management ###

    def _resolve(self, scope: Scope, pkg: Name, name: Name) -> Symbol:
        key = pkg.value
        scope = scope.resolve(key)

        # must be a package
        if not isinstance(scope, PackageScope):
            raise self._error(pkg, '%s is not a package' % repr(key))

        # resolve exported symbols only
        val = name.value
        sym = scope.public.get(name.value)

        # check for resolving result
        if sym is None:
            raise self._error(name, 'unresolved symbol: %s.%s' % (key, val))
        else:
            return sym

    def _declare(self, scope: Scope, name: str, node: Node, symbol: Symbol):
        if not scope.declare(name if name != '_' else BlankGen.next(), symbol):
            raise self._error(node, '%s redeclared in this package' % repr(name))

    ### Package Management ###

    def _parse_tag(self, tag: str) -> Tags:
        ret = Tags(Combinator.AND)
        terms = filter(None, tag.split(','))

        # parse each term
        for term in terms:
            if term[0] != '!':
                ret.tags.append(Tag(term, False))
            else:
                ret.tags.append(Tag(term[1:], True))

        # all done
        return ret

    def _parse_tags(self, fp: TextIO) -> Tuple[Tags, str]:
        tags = Tags(Combinator.AND)
        cache = []
        lines = []

        # read lines from file
        for line in fp:
            ln = line.strip()
            lines.append(line)

            # skip empty lines, commit tags if any
            if not ln:
                tags.tags.extend(map(self._parse_terms, cache))
                cache.clear()
                continue

            # must be line comments
            if not ln.startswith('//'):
                break

            # remove the comment leadings
            ln = ln[2:]
            ln = ln.strip()

            # check for build tags
            if ln.startswith('+build '):
                cache.append(ln)

        # read the remaining content
        lines.append(fp.read())
        return tags, ''.join(lines)

    def _parse_terms(self, line: str) -> Tags:
        ret = Tags(Combinator.OR)
        ret.tags.extend(self._parse_tag(tag) for tag in line.split()[1:])
        return ret

    def _parse_package(self, pkg: str, main: bool) -> Iterable[Package]:
        for name in os.listdir(pkg):
            path = os.path.join(pkg, name)
            base, ext = os.path.splitext(name)

            # file names that begin with "." or "_" are ignored
            if ext != '.go' or base[:1] in ('.', '_') or not os.path.isfile(path):
                continue

            # check for special suffix
            if base.endswith('_test'):
                if not self.test:
                    continue
                else:
                    base = base[:-5]

            # get the architecture
            vals = base.rsplit('_', 2)
            last = vals[-1]

            # calculate which is which
            if last in GOOS:
                osn = last
                arch = ''
            elif last not in GOARCH:
                osn = ''
                arch = ''
            elif len(vals) > 2 and vals[-2] in GOOS:
                osn = vals[-2]
                arch = last
            else:
                osn = ''
                arch = last

            # check for OS name and Arch name
            if osn and osn != self.os or arch and arch != self.arch:
                continue

            # parse the package
            with open(path, newline = None) as fp:
                tags, source = self._parse_tags(fp)

            # make a copy of tags
            tagv = self.tags
            tagv = tagv.copy()

            # add default tags
            tagv.update(GO_VERS)
            tagv.update(GO_EXTRA.get(self.os, []))
            tagv.update([self.os, self.arch, self.backend.value])

            # add "cgo" tag if enabled
            if '%s/%s' % (self.os, self.arch) in CGO_ENABLED:
                tagv.add('cgo')

            # eval tags
            if not tags.eval(tagv):
                continue

            # parse the file
            parser = Parser(Tokenizer(source, path))
            package = parser.parse()

            # selective package filter
            if main or package.name.value != 'main':
                yield package

    ### Type Converters ###

    def _type_of(self, val: Constant) -> Type:
        if val.vt is not None:
            return val.vt
        elif isinstance(val, Int):
            return Types.UntypedInt
        elif isinstance(val, Rune):
            return Types.Int32
        elif isinstance(val, Float):
            return Types.UntypedFloat
        elif isinstance(val, String):
            return Types.UntypedString
        elif isinstance(val, Complex):
            return Types.UntypedComplex
        else:
            raise SystemError('invalid const value')

    def _type_coerce(self, t1: Type, t2: Type) -> Optional[Type]:
        if t1 == t2:
            return t1
        elif t1.kind in COERCING_MAPS.get(t2, {}):
            return t1
        elif t2.kind in COERCING_MAPS.get(t1, {}):
            return t2
        else:
            return None

    ### Operator Appliers ###

    def _cast_to(self, vtype: Type, value: Constant) -> Constant:
        if self._rune_checked(value.vt, vtype):
            return self._make_rune(vtype, value)
        elif self._type_coerce(value.vt, vtype) is not None:
            return self._range_checked(vtype, value, value.value)
        else:
            raise self._error(value, 'invalid type conversion from %s to %s' % (value.vt, vtype))

    def _make_bool(self, val: Node, new: TokenValue) -> Constant:
        ret = Bool(Token(val.col, val.row, val.file, TokenType.Bool, operator.truth(new)))
        ret.vt = Types.UntypedBool
        return ret

    def _make_rune(self, vtype: Type, value: Constant) -> Constant:
        return self._make_typed(value, chr(value.value).encode('utf-8'), vtype)

    def _make_typed(self, val: Node, new: TokenValue, vt: Type) -> Constant:
        vk = CONSTRUCTING_MAPS[vt.kind].kind
        ret = CONSTRUCTING_MAPS[vt.kind](Token(val.col, val.row, val.file, vk, new))
        ret.vt = vt
        return ret

    def _rune_checked(self, rune: Type, string: Type) -> bool:
        return string.kind == Kind.String and (rune is Types.UntypedInt or rune.kind == Kind.Int32)

    def _range_checked(self, vtype: Type, node: Node, value: TokenValue) -> Constant:
        vk = vtype.kind
        val = self._value_casted(vtype, node, value)

        # range checking, untyped values have infinite precision
        if isinstance(vtype, UntypedType):
            return self._make_typed(node, val, vtype)
        elif LITERAL_RANGES[vk](val):
            return self._make_typed(node, val, vtype)
        else:
            raise self._error(node, 'constant %r overflows %s' % (val, vtype))

    def _value_casted(self, vtype: Type, node: Node, value: TokenValue) -> TokenValue:
        if vtype.kind not in CONVERTING_MAPS:
            raise self._error(node, 'invalid constant type %s' % vtype)
        else:
            return CONVERTING_MAPS[vtype.kind](value)

    KindMap = Dict[Kind, TokenType]
    UnaryOps = Callable[[TokenValue], TokenValue]
    BinaryOps = Callable[[TokenValue, TokenValue], TokenValue]

    def _unary_applier(self, val: Constant, op: UnaryOps, kinds: KindMap) -> Constant:
        if val.vt.kind not in kinds:
            raise self._error(val, 'undefined unary operator on type %s' % val.vt)
        elif kinds[val.vt.kind] != val.kind:
            raise SystemError('invalid ast constant kind')
        else:
            return self._range_checked(val.vt, val, op(val.value))

    def _binary_applier(self, lhs: Constant, rhs: Constant, op: BinaryOps, kinds: KindMap) -> Constant:
        t1 = lhs.vt
        t2 = rhs.vt
        tr = self._type_coerce(t1, t2)

        # check for type compatibility
        if tr is None or t1.kind not in kinds or t2.kind not in kinds:
            raise self._error(lhs, 'undefined binary operator between %s and %s' % (t1, t2))

        # apply the operator
        try:
            val = op(lhs.value, rhs.value)
        except ArithmeticError as e:
            val = e

        # check the exception outside of the `except`
        # block, cause we don't want exception chaining here
        if isinstance(val, ArithmeticError):
            raise self._error(rhs, val.args[0])
        else:
            return self._range_checked(tr, lhs, val)

    def _bincmp_applier(self, lhs: Constant, rhs: Constant, op: BinaryOps) -> Constant:
        if self._type_coerce(lhs.vt, rhs.vt) is None:
            raise self._error(lhs, 'undefined binary comparison between %s and %s' % (lhs.vt, rhs.vt))
        else:
            return self._make_bool(lhs, op(lhs.value, rhs.value))

    def _shifts_applier(self, lhs: Constant, rhs: Constant, op: BinaryOps) -> Constant:
        lk = lhs.vt.kind
        rk = rhs.vt.kind

        # check the right operand
        if rk not in INT_KINDS and (rk not in FLOAT_KINDS):
            raise self._error(lhs, 'shift count type %s, must be integer' % rhs.vt)

        # type of the LHS must be ints or untyped types
        if lk not in INT_KINDS and (lk not in FLOAT_KINDS or not isinstance(lhs.vt, UntypedType)):
            raise self._error(lhs, 'invalid shift of type %s' % lhs.vt)

        # split the integer part and fractional part
        lfrac, lval = math.modf(lhs.value)
        rfrac, rval = math.modf(rhs.value)

        # check for LHS fractional part
        if lfrac != 0.0:
            raise self._error(lhs, 'constant %s truncated to integer' % repr(lhs.value))

        # check for RHS fractional part
        if rfrac != 0.0:
            raise self._error(lhs, 'constant %s truncated to integer' % repr(rhs.value))

        # apply the shift
        if lk in INT_KINDS:
            return self._range_checked(lhs.vt, lhs, op(int(lval), int(rval)))
        else:
            return self._range_checked(Types.UntypedInt, lhs, op(int(lval), int(rval)))

    __unary_ops__ = {
        '+': functools.partial(_unary_applier, op = operator.pos  , kinds = NUMERIC_KINDS),
        '-': functools.partial(_unary_applier, op = operator.neg  , kinds = NUMERIC_KINDS),
        '!': functools.partial(_unary_applier, op = operator.not_ , kinds = BOOLEAN_KINDS),
        '^': functools.partial(_unary_applier, op = operator.inv  , kinds = INT_KINDS),
    }

    __binary_ops__ = {
        '+'  : functools.partial(_binary_applier, op = operator.add  , kinds = GENERIC_KINDS),
        '-'  : functools.partial(_binary_applier, op = operator.sub  , kinds = NUMERIC_KINDS),
        '*'  : functools.partial(_binary_applier, op = operator.mul  , kinds = NUMERIC_KINDS),
        '/'  : functools.partial(_binary_applier, op = Ops.div       , kinds = NUMERIC_KINDS),
        '%'  : functools.partial(_binary_applier, op = operator.mod  , kinds = INT_KINDS),
        '&'  : functools.partial(_binary_applier, op = operator.and_ , kinds = INT_KINDS),
        '|'  : functools.partial(_binary_applier, op = operator.or_  , kinds = INT_KINDS),
        '^'  : functools.partial(_binary_applier, op = operator.xor  , kinds = INT_KINDS),
        '&^' : functools.partial(_binary_applier, op = Ops.and_not   , kinds = INT_KINDS),
        '&&' : functools.partial(_binary_applier, op = Ops.bool_and  , kinds = BOOLEAN_KINDS),
        '||' : functools.partial(_binary_applier, op = Ops.bool_or   , kinds = BOOLEAN_KINDS),
        '==' : functools.partial(_bincmp_applier, op = operator.eq),
        '<'  : functools.partial(_bincmp_applier, op = operator.lt),
        '>'  : functools.partial(_bincmp_applier, op = operator.gt),
        '!=' : functools.partial(_bincmp_applier, op = operator.ne),
        '<=' : functools.partial(_bincmp_applier, op = operator.le),
        '>=' : functools.partial(_bincmp_applier, op = operator.ge),
        '<<' : functools.partial(_shifts_applier, op = operator.lshift),
        '>>' : functools.partial(_shifts_applier, op = operator.rshift),
    }

    def _apply_unary(self, val: Constant, op: Operator) -> Constant:
        if op.value not in self.__unary_ops__:
            raise SystemError('invalid unary operator')
        else:
            return self.__unary_ops__[op.value](self, val)

    def _apply_binary(self, lhs: Constant, rhs: Constant, op: Operator) -> Constant:
        if op.value not in self.__binary_ops__:
            raise SystemError('invalid binary operator')
        else:
            return self.__binary_ops__[op.value](self, lhs, rhs)

    ### Expression Evaluators ###

    def _find_val(self, syms: List[Symbol], name: Name) -> Symbol:
        for sym in syms:
            if sym.name == name.value:
                return sym
        else:
            raise SystemError('invalid symbol table')

    def _eval_name(self, ctx: Context, name: Name) -> Constant:
        key = name.value
        sym = ctx.scope.resolve(key)

        # not resolved, maybe defined in another file
        if sym is None and key in ctx.fmap:
            file = ctx.fmap[key]
            rctx = self.Context(ctx.pkg, ctx.fmap, file)

            # find the type specifier
            for spec in file.consts:
                if any(item.value == name.value for item in spec.names):
                    sym = self._find_val(self._infer_const_spec(rctx, spec), name)
                    break

        # still not resolved
        if sym is None:
            if key == 'iota' and self.iota is not None:
                sym = ConstValue('iota', Types.UntypedInt, self.iota)
            else:
                raise self._error(name, 'unresolved constant: %s' % key)

        # must be a constant value
        if not isinstance(sym, ConstValue):
            raise SystemError('invalid const value symbol')
        else:
            return self._make_typed(name, sym.value, sym.type)

    def _eval_args(self, ctx: Context, args: Arguments) -> Constant:
        if len(args.args) != 1:
            raise self._error(args, 'must be a constant expression')
        elif isinstance(args.args[0], Expression):
            return self._eval_expr(ctx, args.args[0])
        elif isinstance(args.args[0], NamedTypeNode):
            return self._eval_type(ctx, args.args[0])
        else:
            raise self._error(args, 'must be a constant expression')

    def _eval_expr(self, ctx: Context, expr: Expression) -> Constant:
        if isinstance(expr.left, Expression):
            lhs = self._eval_expr(ctx, expr.left)
        else:
            lhs = self._eval_primary(ctx, expr.left)

        # no operator is present, must be a single value
        if expr.op is None:
            if expr.right is not None:
                raise SystemError('invalid expr ast')
            else:
                return lhs

        # apply corresponding operators
        if expr.right is None:
            return self._apply_unary(lhs, expr.op)
        else:
            return self._apply_binary(lhs, self._eval_expr(ctx, expr.right), expr.op)

    def _eval_type(self, ctx: Context, vtype: NamedTypeNode) -> Constant:
        name = vtype.name
        package = vtype.package

        # no package specified
        if package is None:
            return self._eval_name(ctx, name)

        # resolve the symbol from package
        scope = ctx.scope
        symbol = self._resolve(scope, package, name)

        # must be constant values
        if not isinstance(symbol, ConstValue):
            raise self._error(name, '%s.%s is not a constant' % (package.value, name.value))
        else:
            return self._make_typed(name, symbol.value, symbol.type)

    def _eval_const(self, _ctx: Context, const: Constant) -> Constant:
        const.vt = self._type_of(const)
        return const

    def _eval_primary(self, ctx: Context, primary: Primary) -> Constant:
        if len(primary.mods) > 2:
            raise self._error(primary, 'must be a constant expression')
        elif not primary.mods:
            return self._eval_singular(ctx, primary)
        else:
            return self._eval_modifiers(ctx, primary)

    def _eval_singular(self, ctx: Context, primary: Primary) -> Constant:
        if isinstance(primary.val, Name):
            return self._eval_name(ctx, primary.val)
        elif isinstance(primary.val, Constant.__args__):
            return self._eval_const(ctx, primary.val)
        elif isinstance(primary.val, Conversion):
            return self._eval_conversion(ctx, primary.val)
        else:
            raise self._error(primary, 'must be a constant expression')

    def _eval_modifiers(self, ctx: Context, primary: Primary) -> Constant:
        val = primary.val
        mods = primary.mods

        # must be a name
        if not isinstance(val, Name):
            raise self._error(primary, 'must be a constant expression')

        # might be a type conversion with package name
        if len(mods) == 2:
            if not isinstance(mods[0], Selector) or \
               not isinstance(mods[1], Arguments):
                raise self._error(primary, 'must be a constant expression')
            else:
                return self._eval_conversion_package(
                    ctx   = ctx,
                    pkg   = val,
                    name  = mods[0].attr,
                    value = self._eval_args(ctx, mods[1]),
                )

        # might be a cross-package identifier reference
        if isinstance(mods[0], Selector):
            name = mods[0].attr
            symbol = self._resolve(ctx.scope, val, name)

            # must be constant values
            if not isinstance(symbol, ConstValue):
                raise self._error(val, '%s.%s is not a constant' % (val.value, name.value))
            else:
                return self._make_typed(val, symbol.value, symbol.type)

        # must be a function call at this point
        if not isinstance(mods[0], Arguments):
            raise self._error(primary, 'must be a constant expression')

        # resolve the function
        mod = mods[0]
        sym = ctx.scope.resolve(primary.val.value)

        # might be a type conversion of a bare name
        if isinstance(sym, Symbols.Type):
            return self._cast_to(sym.type, self._eval_args(ctx, mod))

        # must be the 'complex' function
        if sym is not Functions.Complex:
            raise self._error(primary, 'must be a constant expression')

        # must not be a variadic call
        if mod.var:
            raise self._error(primary, 'must be a constant expression')

        # must call with 2 arguments
        if len(mod.args) != 2:
            raise self._error(primary, '\'complex\' function takes exact 2 arguments')

        # both must all be expressions
        if not all(isinstance(arg, Expression) for arg in mod.args):
            raise self._error(primary, 'must be a constant expression')

        # real and imaginary part
        real = self._eval_expr(ctx, mod.args[0])
        imag = self._eval_expr(ctx, mod.args[1])

        # real part must be ints or floats
        if real.kind not in (TokenType.Int, TokenType.Float):
            raise self._error(real, 'real part of a complex number must be ints or floats')

        # imaginary part must be ints or floats
        if imag.kind not in (TokenType.Int, TokenType.Float):
            raise self._error(real, 'imaginary part of a complex number must be ints or floats')

        # check token value type
        if not isinstance(real.value, (int, float)) or not isinstance(imag.value, (int, float)):
            raise SystemError('invalid token value')

        # infer the result type, and construct the complex value
        vt = self._type_coerce(real.vt, imag.vt)
        val = complex(float(real.value), float(imag.value))

        # check the result type
        if vt is None:
            raise self._error(primary, 'invalid complex component type')

        # build the result
        ret = Complex(Token(primary.col, primary.row, primary.file, TokenType.Complex, val))
        ret.vt = Types.UntypedComplex if isinstance(vt, UntypedType) else Types.Complex128
        return ret

    def _eval_conversion(self, ctx: Context, conversion: Conversion) -> Constant:
        return self._cast_to(
            value = self._eval_expr(ctx, conversion.value),
            vtype = self._infer_type(ctx, conversion.type),
        )

    def _eval_conversion_package(self, ctx: Context, pkg: Name, name: Name, value: Constant) -> Constant:
        scope = ctx.scope
        symbol = self._resolve(scope, pkg, name)

        # check the type, and now it's the same as "simple" conversion
        if not isinstance(symbol, Symbols.Type):
            raise self._error(name, '%s.%s is not a type' % (pkg.value, name.value))
        else:
            return self._cast_to(symbol.type, value)

    ### Type Inferrers ###

    def _infer_type(self, ctx: Context, node: TypeNode) -> Type:
        factory = self.__type_factory__[node.__class__]
        inferrer = self.__type_inferrer__[node.__class__]

        # check for already inferred types
        if node.vt is not None:
            return node.vt

        # special case for type names
        if not factory and not inferrer:
            node.vt = self._infer_type_name(ctx, node)
            return node.vt

        # create an empty type node before inferring recursively
        node.vt = factory(node)
        inferrer(self, ctx, node.vt, node)
        return node.vt

    def _infer_type_name(self, ctx: Context, node: NamedTypeNode) -> Type:
        scope = ctx.scope
        package = node.package

        # types that defined in another package
        if package is not None:
            name = node.name.value
            symbol = self._resolve(scope, package, node.name)

        # types that defined in the current scope chain, or built-in types
        else:
            name = node.name.value
            symbol = scope.resolve(name)

            # still not resolved, maybe defined in another file
            if symbol is None and name in ctx.fmap:
                file = ctx.fmap[name]
                rctx = self.Context(ctx.pkg, ctx.fmap, file)

                # find the type specifier
                for spec in file.types:
                    if spec.name.value == name:
                        symbol = self._infer_type_spec(rctx, spec)
                        break

        # check the resolved type
        if symbol is None:
            raise self._error(node.name, 'unresolved type %s' % repr(name))
        elif not isinstance(symbol, Symbols.Type):
            raise self._error(node.name, 'symbol %s is not a type' % repr(name))
        elif symbol.type is None:
            raise self._error(node.name, 'invalid recursive type alias %s' % repr(name))
        else:
            return symbol.type

    ### Specialized Type Inferrers ###

    def _infer_map_type(self, ctx: Context, vtype: MapType, node: MapTypeNode):
        key = node.key
        ktype = self._infer_type(ctx, key)

        # TODO: check the key type

        # infer the element, and complete the type
        vtype.key = ktype
        vtype.elem = self._infer_type(ctx, node.elem)

    def _infer_array_type(self, ctx: Context, vtype: ArrayType, node: ArrayTypeNode):
        value = self._eval_expr(ctx, node.len)
        etype = self._infer_type(ctx, node.elem)

        # array type must be valid
        if not etype.valid:
            raise self._error(node.elem, 'invalid recursive type')

        # must be an integer expression
        if not isinstance(value, Int):
            raise self._error(node.len, 'must be an integer expression')

        # complete the type
        vtype.len = value.value
        vtype.elem = etype
        vtype.valid = True

    def _infer_slice_type(self, ctx: Context, vtype: SliceType, node: SliceTypeNode):
        elem = node.elem
        vtype.elem = self._infer_type(ctx, elem)

    def _infer_struct_type(self, ctx: Context, vtype: StructType, node: StructTypeNode):
        pass    # TODO: struct type

    def _infer_channel_type(self, ctx: Context, vtype: ChanType, node: ChannelTypeNode):
        vtype.dir = node.dir
        vtype.elem = self._infer_type(ctx, node.elem)

    def _infer_pointer_type(self, ctx: Context, vtype: PtrType, node: PointerTypeNode):
        base = node.base
        vtype.elem = self._infer_type(ctx, base)

    def _infer_vl_array_type(self, ctx: Context, vtype: ArrayType, node: VarArrayTypeNode):
        elem = node.elem
        etype = self._infer_type(ctx, elem)

        # array type must be valid
        if not etype.valid:
            raise self._error(elem, 'invalid recursive type')

        # complete the type
        vtype.elem = etype
        vtype.valid = True

    def _infer_function_type(self, ctx: Context, vtype: FuncType, node: FunctionTypeNode):
        pass    # TODO: function type

    def _infer_interface_type(self, ctx: Context, vtype: InterfaceTypeNode, node: InterfaceTypeNode):
        pass    # TODO: interface type

    __type_factory__ = {
        MapTypeNode       : MapType,
        ArrayTypeNode     : ArrayType,
        SliceTypeNode     : SliceType,
        NamedTypeNode     : None,
        StructTypeNode    : StructType,
        ChannelTypeNode   : ChanType,
        PointerTypeNode   : PtrType,
        VarArrayTypeNode  : ArrayType,
        FunctionTypeNode  : FuncType,
        InterfaceTypeNode : InterfaceType,
    }

    __type_inferrer__ = {
        MapTypeNode       : _infer_map_type,
        ArrayTypeNode     : _infer_array_type,
        SliceTypeNode     : _infer_slice_type,
        NamedTypeNode     : None,
        StructTypeNode    : _infer_struct_type,
        ChannelTypeNode   : _infer_channel_type,
        PointerTypeNode   : _infer_pointer_type,
        VarArrayTypeNode  : _infer_vl_array_type,
        FunctionTypeNode  : _infer_function_type,
        InterfaceTypeNode : _infer_interface_type,
    }

    ### Specification Inferrers ###

    def _infer_var_spec(self, ctx: Context, spec: InitSpec) -> List[Symbol]:
        pass    # TODO: infer var

    def _infer_type_spec(self, ctx: Context, spec: TypeSpec) -> Symbol:
        name = spec.name
        value = name.value

        # wrap the underlying type if it's not an alias
        if spec.alias:
            symbol = Symbols.Type(value, None)
            rstype = symbol
        else:
            rstype = NamedType(value)
            symbol = Symbols.Type(value, rstype)

        # declare the type symbol
        self._declare(ctx.pkg, value, name, symbol)
        rstype.type = self._infer_type(ctx, spec.type)

        # mark the named type valid, if needed
        if not spec.alias:
            rstype.valid = True

        # update the type spec
        spec.vt = rstype.type if spec.alias else rstype
        return symbol

    def _infer_func_spec(self, ctx: Context, spec: Function) -> Symbol:
        pass    # TODO: infer func

    def _infer_const_spec(self, ctx: Context, spec: InitSpec) -> List[Symbol]:
        with self.Iota(self, spec.iota):
            ret = []
            vtype = None

            # parse value type, if any
            if spec.type is not None:
                vtype = self._infer_type(ctx, spec.type)

            # check for value count
            if len(spec.names) != len(spec.values):
                raise self._error(spec, 'expression count mismatch')

            # evaluate all expressions, and resolve each symbol
            for name, value in zip(spec.names, spec.values):
                key = name.value
                val = self._eval_expr(ctx, value)

                # optional type assertion
                if vtype is not None:
                    if self._type_coerce(vtype, val.vt) == vtype:
                        val.vt = vtype
                    else:
                        raise self._error(value, 'invalid constant type %s' % val.vt)

                # check the range
                vv = val.value
                val = self._range_checked(val.vt, value, vv)

                # create a new symbol
                sym = ConstValue(key, val.vt, val.value)
                spec.vt = val.vt

                # declare the symbol
                print('%s %s = %s' % (key, val.vt, val.value))  # FIXME: remove this
                ret.append(sym)
                self._declare(ctx.pkg, key, name, sym)

            # all done
            return ret

    ### Package Inferrers ###

    def _infer_cgo(self, imp: ImportC, pkg: PackageScope):
        # TODO: import `C`
        _ = pkg
        if len(imp.src) <= 64:
            print('* import C :: %s' % repr(imp.src))
        else:
            print('* import C :: %s ...' % repr(imp.src[:64]))

    def _infer_package(
        self,
        main   : bool,
        path   : String,
        trace  : List[str],
        cache  : Dict[bytes, Optional[PackageScope]],
        module : Optional[Module]
    ) -> PackageScope:
        try:
            name = path.value.decode('ascii')
        except UnicodeDecodeError:
            name = None

        # handle this outside of the except block
        # cause we don't want exception chaning here
        if not name or '!' in name:
            raise self._error(path, 'invalid package path: %s' % repr(path.value)[1:])

        # import cycle detection
        if name in trace:
            raise self._error(path, 'import cycle not allowed: %s' % repr(name))

        # resolve the package
        root, fpath = Resolver.lookup(
            name   = name,
            proj   = self.proj,
            root   = self.root,
            paths  = self.paths,
            module = module,
        )

        # check for package
        if root is None and fpath is None:
            raise self._error(path, 'cannot find package %s' % repr(name))

        # find all source files
        files = list(self._parse_package(fpath, main))
        names = sorted(set(file.name.value for file in files))

        # check for source files
        if not names:
            raise self._error(path, 'no source files in %s' % repr(name))

        # must have exact 1 package name
        if len(names) != 1:
            raise self._error(path, 'multiple packages in directory: %s' % ', '.join(names))

        # read "go.mod" in "go mod" mode
        if self.mode == Mode.GO_MOD:
            this = fpath
            fname = None

            # find the "go.mod"
            while this != root:
                fmod = os.path.join(this, 'go.mod')
                this = os.path.dirname(this)

                # found the file
                if os.path.isfile(fmod):
                    fname = fmod
                    break

            # parse the module
            if fname is not None:
                with open(fname, newline = None) as fp:
                    module = Reader().parse(fp.read())

        # check the package name
        if names[0] == '_':
            raise self._error(files[0].name, 'invalid package name')

        # map file names to file objects, and create the meta package
        fmap = {}
        package = PackageScope(names[0], name)

        # phase 1: find out all imported packages
        for file in files:
            imps = file.imports
            syms = package.source(file.file)

            # process every import
            for imp in imps:
                path = imp.path.value
                alias = imp.alias

                # special case of "import `C`"
                if path == b'C' and isinstance(alias, ImportC):
                    self._infer_cgo(imp.alias, package)
                    continue

                # infer dependency recursively, if not done before
                with Trace(trace, name):
                    if path in cache:
                        pkg = cache[path]
                    else:
                        pkg = cache[path] = self._infer_package(
                            main   = False,
                            path   = imp.path,
                            trace  = trace,
                            cache  = cache,
                            module = module,
                        )

                # check for "." import
                if not isinstance(alias, ImportHere):
                    if alias is None:
                        self._declare(syms, pkg.name, imp.path, pkg)
                    else:
                        self._declare(syms, alias.value, alias, pkg)
                else:
                    for key, symbol in pkg.public.items():
                        self._declare(syms, key, alias, symbol)
                    else:
                        self._declare(syms, InplaceGen.next(), alias, pkg)

        # phase 2: map all symbols to files, with duplication check
        for file in files:
            self._map_spec_cv(file, fmap, file.vars)
            self._map_spec_fn(file, fmap, file.funcs)
            self._map_spec_tp(file, fmap, file.types)
            self._map_spec_cv(file, fmap, file.consts)

        # phase 3: infer all types
        for file in files:
            for spec in file.types:
                if spec.vt is None:
                    self._infer_type_spec(self.Context(package, fmap, file), spec)

        # phase 4: infer all constants
        for file in files:
            for spec in file.consts:
                if spec.vt is None:
                    self._infer_const_spec(self.Context(package, fmap, file), spec)

        # phase 5: infer all variables
        for file in files:
            for spec in file.vars:
                if spec.vt is None:
                    self._infer_var_spec(self.Context(package, fmap, file), spec)

        # phase 6: infer all functions
        for file in files:
            for spec in file.funcs:
                if spec.vt is None:
                    self._infer_func_spec(self.Context(package, fmap, file), spec)

        # all done
        package.files = files
        return package

    ### Inferrer Interface ###

    def infer(self, path: str) -> PackageScope:
        return self._infer_package(True, self._string(path), [], {}, None)
