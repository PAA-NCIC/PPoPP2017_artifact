import struct 
import sys
import os
##00
# opbits is input, it is an array
def dump(newcode, mode, arch):
    ##Raw binary
    if mode == 1:
        #print newcode
        #ff = '0x%016x' % newcode 
        base=int(newcode, 16)
        ff="tmp.bin"
        fout = open(ff, 'wb')
        fout.write(struct.pack('<Q', int(base)))
        fout.close()
        #nvdisasm -b  SM35 ff
        #redirect stderr to stdout
        cmd = 'nvdisasm -b %s %s 2>&1' % (arch, ff)
        tmp = os.popen(cmd).read()
        rmfile = 'rm %s' % ff
        os.system(rmfile)
        return tmp
    elif mode == 2:
        if arch == "Maxwell" or arch == "SM52":
            f = open("test_sm52.cubin",'rb+')  
            f.seek(808) 
            base=int(newcode, 16)
            f.write(struct.pack('Q', int(base)))
            f.close()
            cmd = 'cuobjdump --gpu-architecture sm_52 --dump-sass test_sm52.cubin 2>&1'
            tmp = os.popen(cmd).read()
            return tmp
        else: 
            print "You need to provide a cubin template and position of first instruction !"
            exit()
    else:
        print "Error dump mode !"
        exit()

def enumerate(opBits, mode, arch):
    nbits = len(opBits)

    for i in range(0,1<<nbits):
        enc64 = 0
        encBits = i
        for j in range(0, nbits):
            if (encBits >> j & 1 == 1):
                enc64 = enc64 + (1 << opBits[j])
        ff = '0x%016x' % enc64
        ####### dump the code ####
        tmp = dump(ff, mode, arch)
        """
        ff = "binary"
        fout = open(ff, 'wb')
        fout.write(struct.pack('<Q', int(enc64)))
        fout.close() 
        cmd = 'nvdisasm -b SM35 binary 2>&1'
        tmp = os.popen(cmd).read()
        """
        #print tmp
        ##enumerate
        if tmp and tmp.find("?") == -1 and tmp.find("error") == -1:
            instline=tmp.split("\n")
            if (mode == 1):
                inst = instline[1].split();
            else:
                inst = instline[5].split();
            if len(inst) > 4:
                inst.pop(0)
                str = " ".join(inst)
                print str;

if __name__ == "__main__":
    arch = sys.argv[1]
    if arch == "SM35" :
        opBits=[63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 1, 0]
        mode = 1
    elif arch == "SM52":
        opBits=[63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51]
        mode = 2
    else:
        print "Error: you need to run opcode.py to generate opBits first..."
    #print opBits
    enumerate(opBits, mode, arch)
