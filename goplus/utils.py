# -*- coding: utf-8 -*-

import platform

from typing import Any
from typing import Set
from typing import Dict
from typing import Type
from typing import Tuple
from typing import Union
from typing import Callable
from typing import Optional

from types import FunctionType
from bytecode import CompilerFlags

from .assembler import Assembler

class StrictFields(type):
    def __new__(mcs, name: str, bases: Tuple[Type], ns: Dict[str, Any]) -> type:
        noinit = ns.pop('__noinit__', set())
        typing = ns.get('__annotations__', {})
        return super().__new__(mcs, name, bases, _build_attrs(ns, bases, typing, noinit))

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

def _make_init(asm: Assembler, name: str, vtype: Any, real: Any) -> Optional[Callable[[], Any]]:
    if real is bool:
        return lambda: (
            asm.LOAD_CONST(False),
            asm.LOAD_FAST('self'),
            asm.STORE_ATTR(name),
        )
    elif real is dict:
        return lambda: (
            asm.BUILD_MAP(0),
            asm.LOAD_FAST('self'),
            asm.STORE_ATTR(name),
        )
    elif real is list:
        return lambda: (
            asm.BUILD_LIST(0),
            asm.LOAD_FAST('self'),
            asm.STORE_ATTR(name),
        )
    elif real is tuple:
        return lambda: (
            asm.LOAD_CONST(()),
            asm.LOAD_FAST('self'),
            asm.STORE_ATTR(name),
        )
    elif not hasattr(vtype, '__origin__'):
        return None
    elif vtype.__origin__ is not Union:
        return None
    elif bool in vtype.__args__:
        return lambda: (
            asm.LOAD_CONST(False),
            asm.LOAD_FAST('self'),
            asm.STORE_ATTR(name),
        )
    elif type(None) in vtype.__args__:
        return lambda: (
            asm.LOAD_CONST(None),
            asm.LOAD_FAST('self'),
            asm.STORE_ATTR(name),
        )
    else:
        return None

def _build_attrs(attrs: Dict[str, Any], bases: Tuple[Type], fields: Dict[str, Any], noinit: Set[str]) -> Dict[str, Any]:
    asm = Assembler('<compiled>')
    asm.instrs.flags = CompilerFlags.VARARGS | CompilerFlags.VARKEYWORDS
    asm.instrs.argcount = 1
    asm.instrs.argnames = ['self', 'args', 'kwargs']

    # add all attributes
    for name, vtype in fields.items():
        if name not in noinit:
            init = _make_init(asm, name, vtype, _real_type(vtype))
            init and init()

    # set slots and attributes
    attrs['__attrs__'] = set(sorted(attrs.keys()))
    attrs['__slots__'] = tuple(sorted(fields.keys()))

    # create a new `__init__` only when fields present
    if not asm.instrs:
        return attrs

    # check for original `__init__` function
    if '__init__' in attrs:
        orig_init = attrs['__init__']
    elif bases:
        orig_init = bases[0].__init__
    else:
        orig_init = None

    # check for super constructor
    if orig_init is None:
        asm.LOAD_CONST(None)
        asm.RETURN_VALUE()
    else:
        asm.LOAD_CONST(orig_init)
        asm.LOAD_FAST('self')
        asm.LOAD_FAST('args')
        asm.LOAD_FAST('kwargs')
        asm.CALL_FUNCTION_VAR_KW(1)
        asm.RETURN_VALUE()

    # create a new `__init__` function
    attrs['__init__'] = FunctionType(asm.assemble(), {}, '__init__')
    return attrs
