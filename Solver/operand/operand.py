from sets import Set
from subprocess import Popen, PIPE, STDOUT
import subprocess 
import struct
import sys
import os
from collections import defaultdict

types=[]
d=[]

def change(my, origin):
    if (my.op != origin.op):
        return -1
    elif my.modifier != origin.modifier:
        return -2
    elif len(my.operand) != len(origin.operand):
        return -3
    elif my.operandType != origin.operandType:
        return -4
    else:
        for ii in range(len(my.operand)):
            if (my.operand[ii] != origin.operand[ii]):
                return ii
        return -5

class Inst:
    def __init__(self, inst):
        l=len(inst)
        #self.pred=
        self.op = ""
        self.dst=""
        self.src=""
        end = l - 1
        begin = 0
        self.probe= 0
        #check predicate, such as @ 
        if (inst[0].find("@") != -1):
            self.pred=inst[0]
            inst.pop(0)
        #opcode
        op = inst[0].split(".");
        self.op = op[0]
        self.modifier=Set([]);
        op.pop(0)
        if (len(op) >= 1): # has modifiers
            #flag has or not
            self.modifier=Set(op);
        inst.pop(0)
        #R0, [R2], R0;
        str = ' '.join(inst)
        #str.replace(";", ",")
        operands = str.split()
        #print str
        operandType=''
        self.operand=[]
        for operand in operands:
            #check operand type: const? imm? Predicate ? register
            #print operand
            #(ret, value)=self.check(operand)
            ret=self.check(operand)
            operandType = operandType + ret
        self.operandType = operandType
        try:
            idx = d.index(operandType)
        except ValueError:
            if operandType.find("X") == -1:
                d.append(operandType)
                self.probe = 1

    def printInst(self):
        print self.op, self.modifier, self.dst
    def check(self, input):
        operand = input[0:len(input)-1]
        #print operand
        key = operand[0]
        idx = 0
        while operand[idx] == '-' or operand[idx] == '|':
            #idx = idx + 1
            #key = operand[idx]
            operand=operand[1:]
            key=operand[0]
        if key == 'R':
            value=operand[1:]
            try:
                if float(value).is_integer():
                    self.operand.append(value)
                    return ('R')
            except ValueError:
                return 'X'
        elif key == 'P':
            value=operand[1:]
            try:
                if float(value).is_integer():
                    #self. = value
                    self.operand.append(value)
                    return 'P'
            except ValueError:
                return 'X'
        #c[0x0][0x0]
        elif key == 'c':
            value=operand[1:]
            begin=operand.find('x') 
            end=operand.find("]")
            self.operand.append(operand[begin+1:end])
            begin=operand.find('x',end) 
            end=operand.find("]", begin)
            self.operand.append(operand[begin+1:end])
            return 'C'
        #integer 
        else:
            try:
                if float(operand).is_integer():
                    self.operand.append(float(operand))
                    return 'I'
            except ValueError:
                #hex immediate
                if len(operand) >=2 and operand[0:2] == "0x":
                    self.operand.append(operand[1:])
                    return "I"
                else:
                    return 'X'
        return 'X'

            
if __name__ == "__main__":
    count = 0;
    #with open("uuu.sass") as f:
    print "......................................................................."
    print "......R:Register, I:Immediate, M:Memory, P:Predicate, C:constant......."
    print "......Instruction's operands are combinations of R, I, M, P, C........."
    print "......................................................................."
    with open(sys.argv[1]) as f:
        for line in f:
            pos=Set([])
            count += 1
            list=line.split()
            list.pop(0)
            enc = list[len(list)-2]
            base=int(enc, 16)
            list.pop(len(list)-1)
            list.pop(len(list)-1)
            list.pop(len(list)-1)
            origin=Inst(list)

            if origin.probe == 1 and len(origin.operand) > 0:
                pp = [[] for i in range(len(origin.operand)) ]
                for i in range(0, 64):
                    #compute opcode, shift 64-4 bits
                    mask = 2**i
                    newcode = base ^ mask
                    fname = hex(newcode)
                    ff = '0x%016x' % newcode 
                    fout = open(ff, 'wb')
                    fout.write(struct.pack('<Q', int(newcode)))
                    fout.close()
                    #nvdisasm -b  SM35 ff
                    #redirect stderr to stdout
                    cmd = 'nvdisasm -b SM35 %s 2>&1' % ff
                    tmp = os.popen(cmd).read()
                    rmfile = 'rm %s' % ff
                    os.system(rmfile)

                    if tmp and tmp.find("?") == -1 and tmp.find("error") == -1:
                        instline=tmp.split("\n")
                        inst = instline[1].split();
                        inst.pop(0)
                        inst.pop(len(inst) -1)
                        inst.pop(len(inst) -1)
                        inst.pop(len(inst) -1)
                        #ATOM.E.ADD.F32.FTZ.RN R16, [R2], R0;  /* 0x68380000001c0842 */
                        my=Inst(inst)
                        ith=change(my, origin) 
                        if ith >= 0 :
                            pp[ith].append(i)
                print "..........................................................."
                print "(Line", count, "of", sys.argv[1],") operand combination type:", origin.operandType
                for k in range(len(pp)):
                    if k >= len(origin.operand) -1 :
                        operandtype = origin.operandType[len(origin.operand)-1]
                    else:
                        operandtype = origin.operandType[k]
                    print k, "operand is", operandtype 
                    print "Encoding is:", pp[k]
                    print ""
    """
    for dd in d:
        if dd.find("X") == -1 :
            print dd
    """
