# -*- coding: utf-8 -*-

import os
import re
import semver
import datetime
import operator
import itertools

from typing import Dict
from typing import List
from typing import Tuple
from typing import Optional
from typing import Iterable

_HASH_SHA1   = re.compile(r'[0-9a-f]{40}')
_CHAR_ESCAPE = {ord(c): '!' + c.lower() for c in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'}

class Module:
    ver  : str
    name : str
    mods : Dict[str, semver.VersionInfo]

    def __init__(self):
        self.ver = ''
        self.name = ''
        self.mods = {}

    def __repr__(self) -> str:
        return '<Module %s>' % repr(self.name)

class Reader:
    group: Optional[str]

    def __init__(self):
        self.group = None

    def _cast_ver(self, ver: str) -> str:
        if not _HASH_SHA1.match(ver):
            return ver
        else:
            now = datetime.datetime.now()
            return 'v0.0.0-%s-%s' % (now.strftime('%Y%m%d%H%M%S'), ver[:12])

    def _make_ver(self, ver: str, line: int) -> semver.VersionInfo:
        if not ver.startswith('v'):
            raise SyntaxError('invalid semver string at line %d' % line)
        else:
            return semver.parse_version_info(ver[1:])

    def _parse_go(self, mod: Module, val: str, line: int):
        if val == '':
            raise SyntaxError('version required at line %d' % line)
        elif mod.ver != '':
            raise SyntaxError('duplicated version at line %d' % line)
        else:
            mod.ver = val

    def _parse_module(self, mod: Module, val: str, line: int):
        if val == '':
            raise SyntaxError('module name required at line %d' % line)
        elif mod.name != '':
            raise SyntaxError('duplicated module name at line %d' % line)
        else:
            mod.name = val

    def _parse_semver(self, ver: str, line: int) -> semver.VersionInfo:
        return self._make_ver(self._cast_ver(ver), line)

    def _parse_require(self, mod: Module, val: str, line: int):
        self._parse_require_or_exclude(mod, val, line, is_require = True)

    def _parse_exclude(self, mod: Module, val: str, line: int):
        self._parse_require_or_exclude(mod, val, line, is_require = False)

    def _parse_replace(self, mod: Module, val: str, line: int):
        pass  # currently just ignore the "replace" verb

    def _parse_require_or_exclude(self, mod: Module, val: str, line: int, is_require: bool):
        vals = list(filter(None, val.split(' ', 2)))
        path = vals[0]

        # must have at least 2 items
        if len(vals) >= 2:
            ver = self._parse_semver(vals[1], line)
        else:
            raise SyntaxError('invalid package at line %d' % line)

        # check for duplication or existance
        if is_require:
            if path in mod.mods:
                raise SyntaxError('duplicated package at line %d' % line)
        else:
            if path not in mod.mods:
                raise SyntaxError('package not declared at line %d' % line)

        # add or remove the module
        if is_require:
            mod.mods[path] = ver
        elif mod.mods[path] == ver:
            del mod.mods[path]
        else:
            raise SyntaxError('package version mismatch at line %d' % line)

    __verbs__ = {
        'go'      : _parse_go,
        'module'  : _parse_module,
        'require' : _parse_require,
        'exclude' : _parse_exclude,
        'replace' : _parse_replace,
    }

    def parse(self, src: str) -> Module:
        ret = Module()
        lines = map(str.strip, src.splitlines())

        # parse each line
        for i, line in enumerate(lines, 1):
            if not line:
                continue

            # find the verb
            vals = line.split(' ', 1)
            verb = vals[0]

            # group closing
            if verb == ')' and self.group is not None:
                self.group = None
                continue

            # in the middle of a group
            if self.group is not None:
                self.__verbs__[self.group](self, ret, line.lstrip(), i)
                continue

            # check the verb
            if verb not in self.__verbs__:
                raise SyntaxError('invalid verb %r at line %d' % (verb, i))

            # remaining line
            rem = line[len(verb) + 1:]
            rem = rem.lstrip()

            # check for group opening
            if rem == '(':
                self.group = verb
            else:
                self.__verbs__[verb](self, ret, rem, i)

        # all done
        return ret

class Resolver:
    proj   : str
    root   : str
    paths  : List[str]
    module : Optional[Module]

    def __init__(self, proj: str, root: str, paths: List[str], module: Optional[Module] = None):
        self.proj   = proj
        self.root   = root
        self.paths  = paths
        self.module = module

    def _try_sys(self, name: str) -> Iterable[Tuple[str, str]]:
        root = os.path.join(self.root, 'src')
        path = os.path.join(root, name)

        # check for system packages
        if os.path.isdir(path):
            yield root, path

    def _try_vendor(self, name: str) -> Iterable[Tuple[str, str]]:
        if not self.module:
            root = os.path.join(self.proj, 'vendor')
            path = os.path.join(root, name)

            # check for vendored packages
            if os.path.isdir(path):
                yield root, path

    def _try_module(self, name: str) -> Iterable[Tuple[str, str]]:
        if self.module and name in self.module.mods:
            for path in self.paths:
                root = os.path.join(path, 'pkg', 'mod')
                path = os.path.join(root, '%s@v%s' % (
                    name.translate(_CHAR_ESCAPE),
                    self.module.mods[name],
                ))

                # check for modded packages
                if os.path.isdir(path):
                    yield root, path

    def _try_sibling(self, name: str) -> Iterable[Tuple[str, str]]:
        for path in self.paths:
            root = os.path.join(path, 'src')
            path = os.path.join(root, name)

            # check for source packages
            if os.path.isdir(path):
                yield root, path

    def _try_matching(self, name: str) -> Iterable[Tuple[str, str]]:
        for path in self.paths:
            root = os.path.join(path, 'pkg', 'mod')
            path = root

            # traverse each level
            for part in filter(None, name.split(os.sep)):
                dirs = []
                found = False

                # scan the directory
                for item in os.listdir(path):
                    names = item.split('@')
                    fpath = os.path.join(path, item)

                    # match directory names, with file name translation
                    if os.path.isdir(fpath):
                        if names[0] == part.translate(_CHAR_ESCAPE):
                            found = True
                            dirs.append((fpath, '@'.join(names[1:])))

                # nothing matches
                if not found:
                    break

                # find the package that doesn't have a
                # version number, otherwise get the newest version
                for fpath, version in dirs:
                    if not version:
                        path = fpath
                        break
                else:
                    dirs.sort(key = operator.itemgetter(1), reverse = True)
                    path = dirs[0][0]

            # found the required package
            else:
                yield root, path

    Result = Tuple[
        Optional[str],
        Optional[str],
    ]

    def resolve(self, name: str) -> Result:
        return next(itertools.chain(
            self._try_sys(name),
            self._try_vendor(name),
            self._try_module(name),
            self._try_sibling(name),
            self._try_matching(name),
        ), (None, None))

    @classmethod
    def lookup(cls, name: str, proj: str, root: str, paths: List[str], module: Optional[Module] = None) -> Result:
        return cls(proj, root, paths, module).resolve(name)
