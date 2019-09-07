# -*- coding: utf-8 -*-

import types
import opcode
import platform

from bytecode.instr import Instr
from bytecode.instr import Label
from bytecode.bytecode import Bytecode

from bytecode.peephole_opt import ControlFlowGraph
from bytecode.peephole_opt import PeepholeOptimizer

if platform.python_implementation() == 'PyPy':
    from .opstack import stack_effect
    __import__('dis').stack_effect = stack_effect

class Assembler:
    instrs: Bytecode

    def __init__(self, name: str):
        self.instrs = Bytecode()
        self.instrs.name = name

    def __getattr__(self, item) -> 'Instruction':
        return self[item]

    def __getitem__(self, item) -> 'Instruction':
        if item not in opcode.opmap:
            raise ValueError('Invalid instruction: %s' % repr(item))
        else:
            return Instruction(item, self.instrs)

    def label(self) -> 'Location':
        return Location(self.instrs)

    def assemble(self) -> types.CodeType:
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
