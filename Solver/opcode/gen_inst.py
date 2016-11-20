import struct
import sys
##00
for i in range(0, 1024):
    #compute opcode, shift 64-4 bits
    opcode = 2**(64-10)*i
    fname = hex(opcode)
    ff = '0x%016x' % opcode
    print opcode, fname, ff
    fout = open(ff, 'wb')
    fout.write(struct.pack('<Q', int(opcode)))
    fout.close()

##01
for i in range(0, 1024):
    #compute opcode, shift 64-4 bits
    opcode = 2**(64-10)*i + 1
    fname = hex(opcode)
    ff = '0x%016x' % opcode
    print opcode, fname, ff
    fout = open(ff, 'wb')
    fout.write(struct.pack('<Q', int(opcode)))
    fout.close()
##02
for i in range(0, 1024):
    #compute opcode, shift 64-4 bits
    opcode = 2**(64-10)*i + 2
    fname = hex(opcode)
    ff = '0x%016x' % opcode
    print opcode, fname, ff
    fout = open(ff, 'wb')
    fout.write(struct.pack('<Q', int(opcode)))
    fout.close()
##03
for i in range(0, 1024):
    #compute opcode, shift 64-4 bits
    opcode = 2**(64-10)*i + 3
    fname = hex(opcode)
    ff = '0x%016x' % opcode
    print opcode, fname, ff
    fout = open(ff, 'wb')
    fout.write(struct.pack('<Q', int(opcode)))
    fout.close()
