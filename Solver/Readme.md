##Solver is to crack GPU ISA encodings

Output:

* bit position of opcode
* bit positions of operands for different operand type
* bit positions of modifier for each instruction

How to run the workflow?

The workflow is composed of four stage:

* generate PTX code
    * Generate PTX code (.ptx) in ptxgen directory
    * compile PTX to cubin
    * disassembly cubin to sass
    * These sass files are input of the following three solvers
    * Each line of sass files looks like this:
    
    /∗0048∗/ IADD R0, R2, R0; /∗0x4800000000201c03∗/
* opcode solver
    * opcode solver probe 64-bit binary code of sass file by flipping each bit
    and observe whether opcode changes.
    * One of the outputs is bits presents opcode, on Kepler GPU, it should be [63,62,61,60,59,58,57,56,55,54, 1,0]
    
* modifer solver
    * similar to opcode solver
    * The output should be 
* Operand solver
    * R: Register I: Immediate C: constant[][] M: Memory P:Predicate
