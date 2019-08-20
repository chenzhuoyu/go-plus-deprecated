# -*- coding: utf-8 -*-

from typing import Any
from typing import Dict
from typing import Type
from typing import Tuple

class AnnotationSlots(type):
    def __new__(mcs, name: str, bases: Tuple[Type], ns: Dict[str, Any]):
        ns['__slots__'] = tuple(sorted(ns.get('__annotations__', {}).keys()))
        return super().__new__(mcs, name, bases, ns)
