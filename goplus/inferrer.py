# -*- coding: utf-8 -*-

import os
import enum

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
from .ast import Name
from .ast import Node
from .ast import String
from .ast import Package

from .ast import Primary
from .ast import Constant
from .ast import Expression

from .ast import InitSpec
from .ast import TypeSpec
from .ast import Function

from .ast import ImportC
from .ast import ImportHere
from .ast import ImportSpec

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

from .types import Type
from .types import MapType
from .types import PtrType
from .types import ChanType
from .types import FuncType
from .types import ArrayType
from .types import SliceType
from .types import NamedType
from .types import StructType
from .types import InterfaceType

from .symbol import Scope
from .symbol import Symbol
from .symbol import Symbols
from .symbol import PackageScope

from .modules import Module
from .modules import Reader
from .modules import Resolver

from .tokenizer import Token
from .tokenizer import TokenType

from .parser import Parser
from .tokenizer import Tokenizer

# function type of `on_parse(fname: str) -> Package`
OnParseCallback = Callable[
    [str],
    Package,
]

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

PackageMap = Dict[
    str,
    Package,
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
    tags    : Set[str]
    paths   : List[str]
    backend : Backend

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

    ### Symbol Management ###

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

    def _declare(self, scope: Scope, name: str, node: Node, symbol: Symbol):
        if not scope.declare(name if name != '_' else BlankGen.next(), symbol):
            raise self._error(node, '%s redeclared in this package' % repr(name))

    def _resolve_package(self, scope: PackageScope, name: str) -> Optional[Symbol]:
        if name in scope.public:
            return scope.public[name]
        elif name in scope.shared:
            return scope.shared[name]
        else:
            return None

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

            # selectively package filter
            if main or package.name.value != 'main':
                yield package

    ### Expression Reducers ###

    def _reduce_expr(self, expr: Expression) -> Expression:
        pass    # TODO: eval expression

    def _require_const_expr(self, expr: Expression) -> Constant:
        if expr.op or expr.right:
            raise self._error(expr, 'must be a constant expression')
        elif not isinstance(expr.left, Primary):
            raise self._error(expr, 'must be a constant expression')
        elif expr.left.mods:
            raise self._error(expr, 'must be a constant expression')
        elif not isinstance(expr.left.val, Constant.__args__):
            raise self._error(expr, 'must be a constant expression')
        else:
            return expr.left.val

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
            name = package.value
            scope = scope.resolve(name)

            # must be a package
            if not isinstance(scope, PackageScope):
                raise self._error(node.package, '%s is not a package' % repr(name))

            # resolve exported symbols only
            name = node.name.value
            symbol = scope.public.get(name)

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
        etype = self._infer_type(ctx, node.elem)
        value = self._require_const_expr(self._reduce_expr(node.len))

        # array type must be valid
        if not etype.valid:
            raise self._error(node.elem, 'invalid recursive type')

        # must be an integer expression
        if not isinstance(value, Int):
            raise self._error(value, 'must be an integer expression')

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

    def _infer_var_spec(self, ctx: Context, spec: InitSpec) -> Symbol:
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

    def _infer_const_spec(self, ctx: Context, spec: InitSpec) -> Symbol:
        pass    # TODO: infer const

    ### Package Inferrers ###

    def _infer_cgo(self, imp: ImportSpec, pkg: PackageScope):
        # TODO: import `C`
        _ = pkg
        if len(imp.alias.src) <= 64:
            print('* import C :: %s' % repr(imp.alias.src))
        else:
            print('* import C :: %s ...' % repr(imp.alias.src[:64]))

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
                    self._infer_cgo(imp, package)
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
