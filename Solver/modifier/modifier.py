from sets import Set
from subprocess import Popen, PIPE, STDOUT
import subprocess 
import struct
import sys
import os

def and64(enc, bits, pos):
    newenc = enc
    for i in range(len(pos)):
        if bits[i] == 0 :
            base = int("0xFFFFFFFFFFFFFFFF", 16)
            biti = 1 << pos[i]
            biti = base ^  biti # set bit i to 0
            #biti = biti * bits[i]
            newenc = newenc & biti
    return newenc

def or64(enc, bits, pos):
    newenc = enc
    for i in range(len(pos)):
        if bits[i] == 1:
            biti = 1 << pos[i]
            newenc = newenc | biti
    return newenc

class Inst:
    def __init__(self, inst):
        l=len(inst)
        #self.pred=
        self.op = ""
        self.enc=inst[len(inst) -2]
        self.src=""
        begin = 0
        index = begin
        #predicate has or not
        if (inst[index].find("@") != -1):
            self.pred=inst[index]
            inst.pop(0)
        #opcode
        if inst[index][len(inst[index]) - 1] == ";" :
            str=inst[index][0:len(inst[index]) -1 ]
        else:
            str = inst[index]

        op = str.split(".");
        self.op = op[0]
        self.modifier=Set([]);
        op.pop(0)

        ### modifiers ###
        self.modifier=Set(op);

    def printInst(self):
        print self.op, self.modifier


if __name__ == "__main__":
    count = 0;
    with open(sys.argv[1]) as f:
        for line in f:
            pos=[]
            count += 1
            if count == 100:
                break
            #ATOM.E.ADD.F32.FTZ.RN R0, [R2], R0; /* 0x68380000001c0802 */
            list=line.split()
            #list.pop(0)
            origin=Inst(list)
            #origin.printInst()
            enc = list[len(list)-2]
            base=int(enc, 16)
            ### bit by bit xor, to observe whether opcode chages, and guess what this bit represent
            for i in range(0, 64):
                ### compute new opcode ###
                mask = 2**i
                newcode = base ^ mask
                fname = hex(newcode)
                ff = '0x%016x' % newcode 
                fout = open(ff, 'wb')
                fout.write(struct.pack('<Q', int(newcode)))
                fout.close()
                cmd = 'nvdisasm -b SM35 %s 2>&1' % ff
                tmp = os.popen(cmd).read()
                rmfile = 'rm %s' % ff
                os.system(rmfile)

                if tmp and tmp.find("?") == -1 and tmp.find("error") == -1:
                    instline=tmp.split("\n")
                    inst = instline[1].split();
                    inst.pop(0)
                    #ATOM.E.ADD.F32.FTZ.RN R16, [R2], R0;  /* 0x68380000001c0842 */
                    my=Inst(inst)
                    #my.printInst()
                    #print origin.op, my.op, origin.modifier, my.modifier
                    if (my.modifier != origin.modifier) and (my.op == origin.op) :
                        try:
                            xxx = pos.index(i)
                        except ValueError:
                            pos.append(i)

            if (len(pos) > 0):
                print origin.op, "modifier bits:", pos
            #enumerate
            if len(pos) > 0:
                for i in range(1 << len(pos)):
                    bits=[]
                    for j in range(len(pos)): 
                        #expresss i in binary
                        bb= (i >> j ) & 1
                        bits.append(bb)
                    #enumerate value
                    zeros = [0] * len(pos)
                    encpar = int(origin.enc, 16)
                    mm = and64(encpar, zeros, pos)
                    mmm = or64(mm, bits, pos)

                    ff = '0x%016x' % mmm
                    fout = open(ff, 'wb')
                    fout.write(struct.pack('<Q', int(mmm)))
                    fout.close()

                    cmd = 'nvdisasm -b SM35 %s 2>&1' % ff
                    tmp = os.popen(cmd).read()
                    rmfile = 'rm %s' % ff
                    os.system(rmfile)
                    if tmp and tmp.find("?") == -1 and tmp.find("error") == -1:
                        instline=tmp.split("\n")
                        inst = instline[1].split();
                        if (len(inst)>=5):
                            inst.pop(0)
                            str = " ".join(inst)
                            print str;
