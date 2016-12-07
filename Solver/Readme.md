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

Output of opcode solver for SM35 GPU. The following 10 bits represents opcode encoding space of SM35 GPU, then opcode will be generated from these bits.
 

    [63,62,61,60,59,58,57,56,55,54, 1,0]


Partial output of modifier solver for SM35 GPU (all the modifier combinations for each instructions):

    
	MOV modifier bits: [22]
	MOV R1, c[0x0][0x44]; /* 0x64c03c00089c0006 */
	MOV.S R1, c[0x0][0x44]; /* 0x64c03c0008dc0006 */
	LD modifier bits: [22, 55, 56, 57, 58, 59, 60]
	LD.U8 R0, [R4]; /* 0xc0000000001c1000 */
	LD.U8.S R0, [R4]; /* 0xc0000000005c1000 */
	LD.E.U8 R0, [R4]; /* 0xc0800000001c1000 */
	LD.E.U8.S R0, [R4]; /* 0xc0800000005c1000 */
	LD.S8 R0, [R4]; /* 0xc1000000001c1000 */
	LD.S8.S R0, [R4]; /* 0xc1000000005c1000 */
	LD.E.S8 R0, [R4]; /* 0xc1800000001c1000 */
	LD.E.S8.S R0, [R4]; /* 0xc1800000005c1000 */
	LD.U16 R0, [R4]; /* 0xc2000000001c1000 */
	LD.U16.S R0, [R4]; /* 0xc2000000005c1000 */
	LD.E.U16 R0, [R4]; /* 0xc2800000001c1000 */
	LD.E.U16.S R0, [R4]; /* 0xc2800000005c1000 */
	LD.S16 R0, [R4]; /* 0xc3000000001c1000 */
	LD.S16.S R0, [R4]; /* 0xc3000000005c1000 */
	LD.E.S16 R0, [R4]; /* 0xc3800000001c1000 */
	LD.E.S16.S R0, [R4]; /* 0xc3800000005c1000 */
	LD R0, [R4]; /* 0xc4000000001c1000 */
	LD.S R0, [R4]; /* 0xc4000000005c1000 */
	LD.E R0, [R4]; /* 0xc4800000001c1000 */
	LD.E.S R0, [R4]; /* 0xc4800000005c1000 */
	LD.64 R0, [R4]; /* 0xc5000000001c1000 */
	LD.64.S R0, [R4]; /* 0xc5000000005c1000 */
	LD.E.64 R0, [R4]; /* 0xc5800000001c1000 */
	LD.E.64.S R0, [R4]; /* 0xc5800000005c1000 */
	LD.128 R0, [R4]; /* 0xc6000000001c1000 */
	LD.128.S R0, [R4]; /* 0xc6000000005c1000 */
	LD.E.128 R0, [R4]; /* 0xc6800000001c1000 */
	LD.E.128.S R0, [R4]; /* 0xc6800000005c1000 */
	LD.U.128 R0, [R4]; /* 0xc7000000001c1000 */
	LD.U.128.S R0, [R4]; /* 0xc7000000005c1000 */
	LD.E.U.128 R0, [R4]; /* 0xc7800000001c1000 */
	LD.E.U.128.S R0, [R4]; /* 0xc7800000005c1000 */
	LD.CG.U8 R0, [R4]; /* 0xc8000000001c1000 */
	LD.CG.U8.S R0, [R4]; /* 0xc8000000005c1000 */
	LD.E.CG.U8 R0, [R4]; /* 0xc8800000001c1000 */
	LD.E.CG.U8.S R0, [R4]; /* 0xc8800000005c1000 */
	LD.CG.S8 R0, [R4]; /* 0xc9000000001c1000 */
	LD.CG.S8.S R0, [R4]; /* 0xc9000000005c1000 */
	LD.E.CG.S8 R0, [R4]; /* 0xc9800000001c1000 */
	LD.E.CG.S8.S R0, [R4]; /* 0xc9800000005c1000 */
	LD.CG.U16 R0, [R4]; /* 0xca000000001c1000 */
	LD.CG.U16.S R0, [R4]; /* 0xca000000005c1000 */
	LD.E.CG.U16 R0, [R4]; /* 0xca800000001c1000 */
	LD.E.CG.U16.S R0, [R4]; /* 0xca800000005c1000 */
	LD.CG.S16 R0, [R4]; /* 0xcb000000001c1000 */
	LD.CG.S16.S R0, [R4]; /* 0xcb000000005c1000 */
	LD.E.CG.S16 R0, [R4]; /* 0xcb800000001c1000 */
	LD.E.CG.S16.S R0, [R4]; /* 0xcb800000005c1000 */
	LD.CG R0, [R4]; /* 0xcc000000001c1000 */
	LD.CG.S R0, [R4]; /* 0xcc000000005c1000 */
	LD.E.CG R0, [R4]; /* 0xcc800000001c1000 */
	LD.E.CG.S R0, [R4]; /* 0xcc800000005c1000 */
	LD.CG.64 R0, [R4]; /* 0xcd000000001c1000 */
	LD.CG.64.S R0, [R4]; /* 0xcd000000005c1000 */

 
Partial output of operand solver for SM35 GPU is as follows.


    .......................................................................
    ......R:Register, I:Immediate, M:Memory, P:Predicate, C:constant.......
    ......The operands of instruction are combinations of R, I, M, P, C.........
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


