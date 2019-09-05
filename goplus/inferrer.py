# -*- coding: utf-8 -*-

import os

from typing import Set
from typing import Dict
from typing import List
from typing import Callable
from typing import Optional

from .ast import Node
from .ast import String
from .ast import Package

from .ast import ImportC
from .ast import ImportSpec

from .types import MetaPackage

# function type of `on_parse(fname: str) -> Package`
OnParseCallback = Callable[
    [str],
    Package,
]

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
    paths: List[str]
    parse: OnParseCallback

    def __init__(self, parse: OnParseCallback, paths: Optional[List[str]] = None):
        self.parse = parse
        self.paths = paths or []

    def _error(self, node: Node, msg: str) -> SyntaxError:
        return SyntaxError('%s:%d:%d: %s' % (node.file, node.row + 1, node.col + 1, msg))

    def _is_cgo(self, imp: ImportSpec) -> bool:
        return imp.path.value == b'C' and isinstance(imp.alias, ImportC)

    def _find_src(self, name: String) -> List[str]:
        try:
            fname = name.value
            fname = fname.decode('utf-8')
        except UnicodeDecodeError:
            fname = None

        # handle this outside of the except block
        # cause we don't want exception chaning here
        if fname is None:
            raise self._error(name, 'invalid package path: %s' % repr(name.value))

        # construct paths
        paths = filter(os.path.isdir, (
            os.path.join(path, fname)
            for path in self.paths
        ))

        # search for every path
        for path in paths:
            files = filter(os.path.isfile, (
                os.path.join(path, fn)
                for fn in os.listdir(path)
            ))

            # filter out source files
            sources = sorted(
                name
                for name in files
                if name.endswith('.go') and not name.endswith('_test.go')
            )

            # check for files
            if not sources:
                raise self._error(name, 'no source files in %s' % repr(fname))
            else:
                return sources

        # nothing found
        else:
            raise self._error(name, 'cannot find package %s' % repr(fname))

    def infer_package(self, path: String, _trace: Set[str] = None, _cache: Dict[bytes, Optional[MetaPackage]] = None) -> MetaPackage:
        _cache = _cache or {}
        _trace = _trace or set()
        print('* inferring: [%d] %s' % (len(_trace), path.value))

        # import cycle detection
        if path.value in _trace:
            raise self._error(path, 'import cycle not allowed: %s' % repr(path.value.decode('utf-8')))

        # compile all source files
        pkgs = list(map(self.parse, self._find_src(path)))
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
                        print('* import C :: %s' % repr(imp.alias.src))
                    elif imp.path.value not in _cache:
                        _cache[imp.path.value] = None
                        _cache[imp.path.value] = self.infer_package(
                            path   = imp.path,
                            _trace = _trace,
                            _cache = _cache,
                        )

        # all done
        return ret
