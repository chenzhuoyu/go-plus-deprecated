# -*- coding: utf-8 -*-

from enum import IntFlag

class ChannelOptions(IntFlag):
    SEND = 0x01
    RECV = 0x02
    BOTH = SEND | RECV

class FunctionOptions(IntFlag):
    NO_SPLIT  = 0x01
    NO_ESCAPE = 0x02
