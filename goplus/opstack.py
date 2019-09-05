# -*- coding: utf-8 -*-

import sys
import opcode

# OpCode Effects for PyPy 3

class _OpCodes:
    def __getattr__(self, item):
        return opcode.opmap[item]

_FVS_MASK      = 0x4
_FVS_HAVE_SPEC = 0x4

_ops = _OpCodes()
_stack_effect_computers = {}
_static_opcode_stack_effects = {
    _ops.NOP                     : 0,
    _ops.POP_TOP                 : -1,
    _ops.ROT_TWO                 : 0,
    _ops.ROT_THREE               : 0,
    _ops.DUP_TOP                 : 1,
    _ops.DUP_TOP_TWO             : 2,
    _ops.UNARY_POSITIVE          : 0,
    _ops.UNARY_NEGATIVE          : 0,
    _ops.UNARY_NOT               : 0,
    _ops.UNARY_INVERT            : 0,
    _ops.LIST_APPEND             : -1,
    _ops.SET_ADD                 : -1,
    _ops.MAP_ADD                 : -2,
    _ops.BINARY_POWER            : -1,
    _ops.BINARY_MULTIPLY         : -1,
    _ops.BINARY_MODULO           : -1,
    _ops.BINARY_ADD              : -1,
    _ops.BINARY_SUBTRACT         : -1,
    _ops.BINARY_SUBSCR           : -1,
    _ops.BINARY_FLOOR_DIVIDE     : -1,
    _ops.BINARY_TRUE_DIVIDE      : -1,
    _ops.BINARY_MATRIX_MULTIPLY  : -1,
    _ops.BINARY_LSHIFT           : -1,
    _ops.BINARY_RSHIFT           : -1,
    _ops.BINARY_AND              : -1,
    _ops.BINARY_OR               : -1,
    _ops.BINARY_XOR              : -1,
    _ops.INPLACE_FLOOR_DIVIDE    : -1,
    _ops.INPLACE_TRUE_DIVIDE     : -1,
    _ops.INPLACE_ADD             : -1,
    _ops.INPLACE_SUBTRACT        : -1,
    _ops.INPLACE_MULTIPLY        : -1,
    _ops.INPLACE_MODULO          : -1,
    _ops.INPLACE_POWER           : -1,
    _ops.INPLACE_MATRIX_MULTIPLY : -1,
    _ops.INPLACE_LSHIFT          : -1,
    _ops.INPLACE_RSHIFT          : -1,
    _ops.INPLACE_AND             : -1,
    _ops.INPLACE_OR              : -1,
    _ops.INPLACE_XOR             : -1,
    _ops.STORE_SUBSCR            : -3,
    _ops.DELETE_SUBSCR           : -2,
    _ops.GET_ITER                : 0,
    _ops.FOR_ITER                : 1,
    _ops.BREAK_LOOP              : 0,
    _ops.CONTINUE_LOOP           : 0,
    _ops.SETUP_LOOP              : 0,
    _ops.PRINT_EXPR              : -1,
    _ops.WITH_CLEANUP_START      : 0,
    _ops.WITH_CLEANUP_FINISH     : -1,
    _ops.LOAD_BUILD_CLASS        : 1,
    _ops.POP_BLOCK               : 0,
    _ops.POP_EXCEPT              : -1,
    _ops.END_FINALLY             : -4,
    _ops.SETUP_WITH              : 1,
    _ops.SETUP_FINALLY           : 0,
    _ops.SETUP_EXCEPT            : 0,
    _ops.RETURN_VALUE            : -1,
    _ops.YIELD_VALUE             : 0,
    _ops.YIELD_FROM              : -1,
    _ops.COMPARE_OP              : -1,
    _ops.LOOKUP_METHOD           : 1,
    _ops.LOAD_NAME               : 1,
    _ops.STORE_NAME              : -1,
    _ops.DELETE_NAME             : 0,
    _ops.LOAD_FAST               : 1,
    _ops.STORE_FAST              : -1,
    _ops.DELETE_FAST             : 0,
    _ops.LOAD_ATTR               : 0,
    _ops.STORE_ATTR              : -2,
    _ops.DELETE_ATTR             : -1,
    _ops.LOAD_GLOBAL             : 1,
    _ops.STORE_GLOBAL            : -1,
    _ops.DELETE_GLOBAL           : 0,
    _ops.LOAD_CLOSURE            : 1,
    _ops.LOAD_DEREF              : 1,
    _ops.STORE_DEREF             : -1,
    _ops.DELETE_DEREF            : 0,
    _ops.GET_AWAITABLE           : 0,
    _ops.SETUP_ASYNC_WITH        : 0,
    _ops.BEFORE_ASYNC_WITH       : 1,
    _ops.GET_AITER               : 0,
    _ops.GET_ANEXT               : 1,
    _ops.GET_YIELD_FROM_ITER     : 0,
    _ops.LOAD_CONST              : 1,
    _ops.IMPORT_STAR             : -1,
    _ops.IMPORT_NAME             : -1,
    _ops.IMPORT_FROM             : 1,
    _ops.JUMP_FORWARD            : 0,
    _ops.JUMP_ABSOLUTE           : 0,
    _ops.JUMP_IF_TRUE_OR_POP     : 0,
    _ops.JUMP_IF_FALSE_OR_POP    : 0,
    _ops.POP_JUMP_IF_TRUE        : -1,
    _ops.POP_JUMP_IF_FALSE       : -1,
    _ops.JUMP_IF_NOT_DEBUG       : 0,
    _ops.BUILD_LIST_FROM_ARG     : 1,
    _ops.LOAD_CLASSDEREF         : 1,
}

if sys.version_info >= (3, 6, 0):
    _static_opcode_stack_effects[_ops.STORE_ANNOTATION] = -1
    _static_opcode_stack_effects[_ops.SETUP_ANNOTATIONS] = 0

def _num_args(oparg):
    return (oparg % 256) + 2 * ((oparg // 256) % 256)

def _compute_UNPACK_SEQUENCE(arg):
    return arg - 1

def _compute_UNPACK_EX(arg):
    return (arg & 0xff) + (arg >> 8)

def _compute_BUILD_TUPLE(arg):
    return 1 - arg

def _compute_BUILD_TUPLE_UNPACK(arg):
    return 1 - arg

def _compute_BUILD_LIST(arg):
    return 1 - arg

def _compute_BUILD_LIST_UNPACK(arg):
    return 1 - arg

def _compute_BUILD_SET(arg):
    return 1 - arg

def _compute_BUILD_SET_UNPACK(arg):
    return 1 - arg

def _compute_BUILD_MAP(arg):
    return 1 - 2 * arg

def _compute_BUILD_MAP_UNPACK(arg):
    return 1 - arg

def _compute_BUILD_MAP_UNPACK_WITH_CALL(arg):
    return 1 - (arg & 0xff)

def _compute_MAKE_CLOSURE(arg):
    return -2 - _num_args(arg) - ((arg >> 16) & 0xffff)

def _compute_MAKE_FUNCTION(arg):
    return -1 - _num_args(arg) - ((arg >> 16) & 0xffff)

def _compute_BUILD_SLICE(arg):
    if arg == 3:
        return -2
    else:
        return -1

def _compute_RAISE_VARARGS(arg):
    return -arg

def _compute_CALL_FUNCTION(arg):
    return -_num_args(arg)

def _compute_CALL_FUNCTION_VAR(arg):
    return -_num_args(arg) - 1

def _compute_CALL_FUNCTION_KW(arg):
    return -_num_args(arg) - 1

def _compute_CALL_FUNCTION_VAR_KW(arg):
    return -_num_args(arg) - 2

if sys.version_info >= (3, 7, 0):
    def _compute_CALL_METHOD(arg):
        return -_num_args(arg) - 1

if sys.version_info >= (3, 6, 0):
    def _compute_FORMAT_VALUE(arg):
        if (arg & _FVS_MASK) != _FVS_HAVE_SPEC:
            return 0
        else:
            return -1

if sys.version_info >= (3, 6, 0):
    def _compute_BUILD_STRING(arg):
        return 1 - arg

class _Dummy:
    for name, func in globals().items():
        if name.startswith('_compute_'):
            _stack_effect_computers[getattr(_ops, name[9:])] = func

    for op, value in _static_opcode_stack_effects.items():
        _stack_effect_computers[op] = lambda _, val = value: val

def stack_effect(op, arg):
    try:
        return _static_opcode_stack_effects[op]
    except KeyError:
        try:
            return _stack_effect_computers[op](arg)
        except KeyError:
            raise KeyError('Unknown stack effect for %s (%s)' % (opcode.opname[op], op))
