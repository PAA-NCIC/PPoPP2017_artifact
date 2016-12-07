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
    * sample output

    .......................................................................
    ......R:Register, I:Immediate, M:Memory, P:Predicate, C:constant.......
    ......Instruction's operands are combinations of R, I, M, P, C.........
    .......................................................................
    argv[1]: disasssembly file;
    argv[2]: arch: SM21|SM35|Maxwell|Kepler|SM52 
    .......................................................... 
    (Line 5 of opcode.sass ) operand combination type: I
    0 operand is I
    Encoding is: [23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54]
    ...........................................................
    (Line 22 of opcode.sass ) operand combination type: R
    0 operand is R
    Encoding is: [2, 3, 4, 5, 6, 7, 8, 9]
    ...........................................................
    (Line 530 of opcode.sass ) operand combination type: RRII
    0 operand is R
    Encoding is: [2, 3, 4, 5, 6, 7, 8, 9]
    1 operand is R
    Encoding is: [10, 11, 12, 13, 14, 15, 16, 17]
    2 operand is I
    Encoding is: [23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54]
    3 operand is I
    Encoding is: [56, 57, 58, 59, 60]
    ...........................................................
    (Line 1250 of opcode.sass ) operand combination type: RRIP
    0 operand is R
    Encoding is: [2, 3, 4, 5, 6, 7, 8, 9]
    1 operand is R
    Encoding is: [10, 11, 12, 13, 14, 15, 16, 17]
    2 operand is I
    Encoding is: [41]
    3 operand is P
    Encoding is: [42, 43, 44]
