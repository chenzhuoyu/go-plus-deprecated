# -*- coding: utf-8 -*-

import opcode
import platform

from types import CodeType
from typing import Set

from bytecode.instr import Instr
from bytecode.instr import Label
from bytecode.instr import Compare
from bytecode.bytecode import Bytecode

from bytecode.peephole_opt import ControlFlowGraph
from bytecode.peephole_opt import PeepholeOptimizer

if platform.python_implementation() == 'PyPy':
    from .opstack import stack_effect
    __import__('dis').stack_effect = stack_effect

_COMPARE_MAP = {
    '=='  : Compare.EQ,
    '>'   : Compare.GT,
    '<'   : Compare.LT,
    '!='  : Compare.NE,
    '>='  : Compare.GE,
    '<='  : Compare.LE,
    'exc' : Compare.EXC_MATCH,
}

class Assembler:
    names  : Set[str]
    locals : Set[str]
    instrs : Bytecode

    def __init__(self, name: str):
        self.names = set()
        self.locals = set()
        self.instrs = Bytecode()
        self.instrs.name = name

    def __getattr__(self, item) -> 'Instruction':
        return self[item]

    def __getitem__(self, item) -> 'Instruction':
        if item not in opcode.opmap:
            raise ValueError('Invalid instruction: %s' % repr(item))
        else:
            return Instruction(item, self.instrs)

    def store(self, name: str):
        self.locals.add(name)
        return self.STORE_FAST(name)

    def label(self) -> 'Location':
        return Location(self.instrs)

    def compare(self, op: str):
        if op not in _COMPARE_MAP:
            raise ValueError('Invalid comparison operator: %s' % repr(op))
        else:
            return self.COMPARE_OP(_COMPARE_MAP[op])

    def assemble(self) -> CodeType:
        cfg = ControlFlowGraph.from_bytecode(self.instrs)
        PeepholeOptimizer().optimize_cfg(cfg)
        return cfg.to_bytecode().to_code()

class Location:
    label  : Label
    instrs : Bytecode

    def __init__(self, instrs: Bytecode):
        self.label = Label()
        self.instrs = instrs

    @property
    def pc(self):
        return self.label

    def commit(self):
        self.instrs.append(self.label)
        return self.label

class Instruction:
    name   : str
    instrs : Bytecode

    def __init__(self, name: str, instrs: Bytecode):
        self.name = name
        self.instrs = instrs

    def __call__(self, *args, **kwargs) -> Instr:
        ret = Instr(self.name, *args, **kwargs)
        self.instrs.append(ret)
        return ret
