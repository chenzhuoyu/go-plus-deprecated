# -*- coding: utf-8 -*-

import os
import enum
import itertools

from typing import Set
from typing import Dict
from typing import List
from typing import Tuple
from typing import Union
from typing import TextIO
from typing import Callable
from typing import Iterable
from typing import Optional

from .ast import Node
from .ast import String
from .ast import Package

from .ast import ImportC
from .ast import ImportSpec

from .types import MetaPackage

from .modules import Module
from .modules import Reader
from .modules import Resolver

from .parser import Parser
from .tokenizer import Tokenizer

from .tokenizer import Token
from .tokenizer import TokenType

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

    ### Inferrer Stages ###

    def _infer_cgo(self, imp: ImportSpec):
        if len(imp.alias.src) <= 64:
            print('* import C :: %s' % repr(imp.alias.src))
        else:
            print('* import C :: %s ...' % repr(imp.alias.src[:64]))

    def _infer_package(
        self,
        main   : bool,
        path   : String,
        trace  : List[str],
        cache  : Dict[bytes, Optional[MetaPackage]],
        module : Optional[Module]
    ) -> MetaPackage:
        try:
            name = path.value.decode('ascii')
        except UnicodeDecodeError:
            name = None

        # handle this outside of the except block
        # cause we don't want exception chaning here
        if not name or '!' in name:
            raise self._error(path, 'invalid package path: %s' % repr(path.value)[1:])

        print('* inferring: [%r] %s' % (len(trace), name))

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
        pkgs = list(self._parse_package(fpath, main))
        names = sorted(set(pkg.name.value for pkg in pkgs))

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

        # create the meta package
        pkg = names[0]
        ret = MetaPackage(pkg)

        # flatten all imports
        imps = itertools.chain.from_iterable(
            pkg.imports
            for pkg in pkgs
        )

        # infer imported packages first
        for imp in imps:
            path = imp.path.value
            alias = imp.alias

            # already inferred before, skip it
            if path in cache:
                continue

            # special case of "import `C`"
            if path == b'C' and isinstance(alias, ImportC):
                self._infer_cgo(imp)
                continue

            # infer dependency recursively
            with Trace(trace, name):
                cache[path] = self._infer_package(
                    main   = False,
                    path   = imp.path,
                    trace  = trace,
                    cache  = cache,
                    module = module,
                )

        # TODO: infer everything else

        # all done
        return ret

    ### Inferrer Interface ###

    def infer(self, path: str) -> MetaPackage:
        return self._infer_package(True, self._string(path), [], {}, None)
