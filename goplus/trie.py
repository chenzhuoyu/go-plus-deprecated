# -*- coding: utf-8 -*-

from typing import Dict
from typing import Iterable
from collections import defaultdict

class Node:
    is_leaf  : bool
    children : Dict[str, 'Node']

    def __init__(self):
        self.is_leaf = False
        self.children = defaultdict(self.__class__)

    def __contains__(self, item) -> bool:
        return item in self.children

    def insert(self, val: str):
        insert_into(self, val)

    def insert_many(self, vals: Iterable[str]):
        for val in vals:
            insert_into(self, val)

def build_from(items: Iterable[str]) -> Node:
    root = Node()
    root.insert_many(items)
    return root

def insert_into(node: Node, item: str):
    for ch in item:
        node = node.children[ch]
    else:
        node.is_leaf = True
