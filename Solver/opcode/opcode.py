from sets import Set
from subprocess import Popen, PIPE, STDOUT
import subprocess 
import struct
import sys
import os

class Inst:
    ### parse instruction to structure###
    ### inst is a list like this ["MOV", "R1,", "c[0x0][0x44];", "/*","0x64c03c00089c0006","*/"]
    def __init__(self, inst):
        l=len(inst)
        self.op = ""
        index = 0
        #check predicate, such as @P0
        if (inst[index].find("@") != -1):
            self.pred=inst[index]
            inst.pop(0) #pop out predicate
        #check opcode
        if inst[index][len(inst[index]) - 1] == ";" :
            str=inst[index][0:len(inst[index]) -1 ]
        else:
            str = inst[index]
        op = str.split(".");
        self.op = op[0]

    def printInst(self):
        print self.op


if __name__ == "__main__":
    count = 0;
    pos=Set([])
    #with open("sm35.sass") as f:
    with open(sys.argv[1]) as f:
        for line in f:
            count += 1
            if count == 100:
                break
            #ATOM.E.ADD.F32.FTZ.RN R0, [R2], R0; /* 0x68380000001c0802 */
            list=line.split()
            ## the origininal instruction structure
            origin=Inst(list)
            ##find the 64-bit encodings
            enc = list[len(list)-2]
            base=int(enc, 16)
            ### bit by bit xor, to observe whether opcode chages, and guess what this bit represent
            for i in range(0, 64):
                ##################### compute opcode, shift 64-4 bits ###########
                mask = 2**i
                newcode = base ^ mask
                fname = hex(newcode)
                #################### disassemble the new code ##################
                ff = '0x%016x' % newcode 
                fout = open(ff, 'wb')
                fout.write(struct.pack('<Q', int(newcode)))
                fout.close()
                cmd = 'nvdisasm -b SM35 %s 2>&1' % ff
                tmp = os.popen(cmd).read()
                rmfile = 'rm %s' % ff
                os.system(rmfile)

                ### compare the disassemble to check which field changes: opcode, operand or modifer ###
                if tmp and tmp.find("?") == -1 and tmp.find("error") == -1:
                    instline=tmp.split("\n")
                    inst = instline[1].split();
                    inst.pop(0)
                    #### parse the new generated disassembly ##
                    my=Inst(inst)
                    ### if opcode is changed, then this bit represent opcode, we find it!!! ###
                    if my.op != origin.op and len(inst) > 3:
                        print "opcode changes: ", origin.op,"=>", my.op, "when bit [", i, "]is flipped"
                        pos.add(i)
    print ""
    print "Done: found out bits represents opcode on Kepler...."
    print "Opcodes positons:", pos
    print "Then we could enumate in the space...."
