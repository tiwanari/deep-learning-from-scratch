#!/usr/bin/env python3
from and_gate import AND
from nand_gate import NAND
from or_gate import OR


def XOR(x1, x2):
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    return AND(s1, s2)
