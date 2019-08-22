# -*- coding: utf-8 -*-

from typing import Any
from typing import Dict
from typing import Type
from typing import Tuple
from typing import Union

class StrictFields(type):
    def __new__(mcs, name: str, bases: Tuple[Type], ns: Dict[str, Any]):
        typing = ns.get('__annotations__', {})
        return super().__new__(mcs, name, bases, _build_attrs(ns, bases, typing))

# noinspection PyProtectedMember
def _add_attrs(self: Any, felds: Dict[str, Any]):
    for name, vtype in felds.items():
        if vtype is bool:
            setattr(self, name, False)
        elif not hasattr(vtype, '_name'):
            continue
        elif vtype._name == 'Dict':
            setattr(self, name, {})
        elif vtype._name == 'List':
            setattr(self, name, [])
        elif vtype._name == 'Tuple':
            setattr(self, name, ())
        elif vtype.__origin__ is Union:
            _add_union(self, name, vtype)

def _add_union(self: Any, name: str, vtype: Any):
    if bool in vtype.__args__:
        setattr(self, name, False)
    elif type(None) in vtype.__args__:
        setattr(self, name, None)

def _build_attrs(ns: Dict[str, Any], bases: Tuple[Type], fields: Dict[str, Any]) -> Dict[str, Any]:
    def init(self, *args, **kwargs):
        _add_attrs(self, fields)
        orig_init and orig_init(self, *args, **kwargs)

    # check for original `__init__` function
    if '__init__' in ns:
        orig_init = ns['__init__']
    elif bases:
        orig_init = bases[0].__init__
    else:
        orig_init = None

    # create a new `__init__` function and `__slots__`
    ns['__init__'] = init
    ns['__slots__'] = tuple(sorted(fields.keys()))
    return ns
