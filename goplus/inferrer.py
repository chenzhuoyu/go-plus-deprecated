# -*- coding: utf-8 -*-

import os

from typing import Set
from typing import Dict
from typing import List
from typing import Tuple
from typing import Callable
from typing import Iterable
from typing import Optional

from .ast import Node
from .ast import String
from .ast import Package

from .ast import ImportC
from .ast import ImportSpec

from .modules import Module
from .modules import Reader
from .modules import Resolver

from .types import MetaPackage

# function type of `on_parse(fname: str) -> Package`
OnParseCallback = Callable[
    [str],
    Package,
]

GOOS = {
    'android',
    'darwin',
    'dragonfly',
    'freebsd',
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
    's390',
    's390x',
    'sparc',
    'sparc64',
}

class Trace:
    path  : str
    trace : Set[str]

    def __init__(self, trace: Set[str], path: String):
        self.path = path.value
        self.trace = trace

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.trace.remove(self.path)

    def __enter__(self):
        self.trace.add(self.path)
        return self

class Inferrer:
    os       : str
    arch     : str
    test     : bool
    parse    : OnParseCallback
    resolver : Resolver

    def __init__(self, osn: str, arch: str, is_test: bool, parse: OnParseCallback, resolver: Resolver):
        self.os       = osn
        self.arch     = arch
        self.test     = is_test
        self.parse    = parse
        self.resolver = resolver

    def _error(self, node: Node, msg: str) -> SyntaxError:
        return SyntaxError('%s:%d:%d: %s' % (node.file, node.row + 1, node.col + 1, msg))

    def _is_cgo(self, imp: ImportSpec) -> bool:
        return imp.path.value == b'C' and isinstance(imp.alias, ImportC)

    def _list_src(self, pkg: str) -> Iterable[str]:
        for name in os.listdir(pkg):
            path = os.path.join(pkg, name)
            base, ext = os.path.splitext(name)

            # only '.go' files
            if ext != '.go' or not os.path.isfile(path):
                continue

            # check for "_test" suffix
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

            # yield the full file name if it maches the OS and Arch
            if osn in ('', self.os) and arch in ('', self.arch):
                yield os.path.join(pkg, name)

    def _find_src(self, name: String, module: Module) -> Tuple[List[str], Optional[Module]]:
        try:
            fname = name.value
            fname = fname.decode('utf-8')
        except UnicodeDecodeError:
            fname = None

        # handle this outside of the except block
        # cause we don't want exception chaning here
        if fname is None:
            raise self._error(name, 'invalid package path: %s' % repr(name.value))

        # resolve package
        root, package = self.resolver.resolve(
            name   = fname,
            module = module,
        )

        # check for package
        if package is None:
            raise self._error(name, 'cannot find package %s' % repr(fname))

        # filter out source files
        mod = None
        modf = os.path.join(package, 'go.mod')
        files = sorted(self._list_src(package))

        # parse the module if any
        # TODO: search until `root`
        if os.path.isfile(modf):
            with open(modf, newline = None) as fp:
                mod = Reader().parse(fp.read())

        # check for files
        if not files:
            raise self._error(name, 'no source files in %s' % repr(fname))
        else:
            return files, mod

    def infer_package(
        self,
        path   : String,
        module : Optional[Module],
        _trace : Set[str] = None,
        _cache : Dict[bytes, Optional[MetaPackage]] = None
    ) -> MetaPackage:
        _cache = _cache or {}
        _trace = _trace or set()
        print('* inferring: [%d] %s' % (len(_trace), path.value))

        # import cycle detection
        if path.value in _trace:
            raise self._error(path, 'import cycle not allowed: %s' % repr(path.value.decode('utf-8')))

        # find all source files
        files, module = self._find_src(
            name   = path,
            module = module,
        )

        # compile all source files
        pkgs = list(map(self.parse, files))
        names = sorted(set(pkg.name.value for pkg in pkgs))

        # special case for 'main' package
        if len(names) > 1 and 'main' in names:
            names.remove('main')

        # check for package names
        if len(names) != 1:
            raise self._error(path, 'multiple packages in directory: %s' % ', '.join(names))

        # create the meta package
        pkg = names[0]
        ret = MetaPackage(pkg)

        # infer imported packages first
        with Trace(_trace, path):
            for pkg in pkgs:
                for imp in pkg.imports:
                    if self._is_cgo(imp):
                        if len(imp.alias.src) <= 64:
                            print('* import C :: %s' % repr(imp.alias.src))
                        else:
                            print('* import C :: %s ...' % repr(imp.alias.src[:64]))
                    elif imp.path.value not in _cache:
                        _cache[imp.path.value] = None
                        _cache[imp.path.value] = self.infer_package(
                            path   = imp.path,
                            module = module,
                            _trace = _trace,
                            _cache = _cache,
                        )

        # all done
        return ret
