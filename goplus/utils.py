# -*- coding: utf-8 -*-

import platform

from typing import Any
from typing import Dict
from typing import Type
from typing import Tuple
from typing import Union

class StrictFields(type):
    def __new__(mcs, name: str, bases: Tuple[Type], ns: Dict[str, Any]):
        typing = ns.get('__annotations__', {})
        return super().__new__(mcs, name, bases, _build_attrs(ns, bases, typing))

# this differs between implementations
if platform.python_implementation() == 'PyPy':
    def _real_type(vtype: Any) -> type:
        try:
            return vtype.__extra__
        except AttributeError:
            return vtype
else:
    def _real_type(vtype: Any) -> type:
        try:
            return vtype.__origin__
        except AttributeError:
            return vtype

def _add_attrs(self: Any, fields: Dict[str, Any]):
    for name, vtype in fields.items():
        _add_generic(self, name, vtype, _real_type(vtype))

def _add_union(self: Any, name: str, vtype: Any):
    if bool in vtype.__args__:
        setattr(self, name, False)
    elif type(None) in vtype.__args__:
        setattr(self, name, None)

def _add_generic(self: Any, name: str, vtype: Any, real: Any):
    try:
        if real is bool:
            setattr(self, name, False)
        elif real is dict:
            setattr(self, name, {})
        elif real is list:
            setattr(self, name, [])
        elif real is tuple:
            setattr(self, name, ())
        elif vtype.__origin__ is Union:
            _add_union(self, name, vtype)
    except AttributeError:
        pass

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
