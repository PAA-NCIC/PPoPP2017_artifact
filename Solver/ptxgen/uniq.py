from sets import Set
from subprocess import Popen, PIPE, STDOUT
import subprocess 
import struct
import sys
import os

class Inst:
    def __init__(self, inst):
        l=len(inst)
        self.op = ""
        self.dst=""
        self.src=""

        #check predicate field: @P0
        if (inst[0].find("@") != -1):
            self.pred=inst[0]
            inst.pop(0) # remove the predict

        #check 0 operand field such as "RRO;", then remove semicolon
        index = 0
        if inst[index][len(inst[index]) - 1] == ";" :
            str=inst[index][0:len(inst[index]) -1 ]
        else:
            str = inst[index]
        #opcode
        op = str.split(".");
        self.op = op[0]

    def printInst(self):
        print self.op


if __name__ == "__main__":
    reg=dict()
    const=dict()
    imm=dict()
    opset=Set([])
    #with open("sm21_uniq.sass") as f:
    with open(sys.argv[1]) as f:
        for line in f:
            field = line.split()
            my=Inst(field)
            if not my.op in opset:
                opset.add(my.op)
                sys.stdout.write(line)
