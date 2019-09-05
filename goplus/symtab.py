# -*- coding: utf-8 -*-

from typing import Dict
from typing import Optional

from .types import Type
from .utils import StrictFields

class Symbol(metaclass = StrictFields):
    name: str
    type: Type

class SymScope:
    syms   : Dict[str, Symbol]
    parent : Optional['SymScope']

    def fork(self) -> 'SymScope':
        ret = SymScope()
        ret.parent = self
        return ret

    def lookup(self, name: str) -> Optional[Symbol]:
        if name in self.syms:
            return self.syms[name]
        elif self.parent is not None:
            return self.parent.lookup(name)
        else:
            return None
