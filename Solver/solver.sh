#generate ptx file
<<BLOCK
mkdir -p ptxgen/data/ptx
echo "generating sass files in ptxgen/data/ptx directory..."
echo ".................................................................."
#ptxgen/ptxgen -a=sm_35

ptx=`ls ptxgen/data/ptx/*.ptx`

################ generate cubin ##############
mkdir -p ptxgen/data/cubin
echo "compile .ptx file to cubin files in ptxgen/data/cubin directory..."
echo "It may take serveral miniutes..."
echo ".................................................................."
for p in $ptx
do
     f=`echo $p | cut -d / -f 4 |cut -d . -f 1` 
     fout="ptxgen/data/cubin/"$f".cubin"
     #echo $fout
     ptxas -arch sm_35 -m 64 $p -o $fout
done

########## disassembly it using cubin ########
echo "disassble .cubin file to sass files in ptxgen/data/asm directory..."
echo "It may take serveral miniutes..."
echo ".................................................................."
cubin=`ls ptxgen/data/cubin/*.cubin`
mkdir -p ptxgen/data/asm
for p in $cubin
do
     f=`echo $p | cut -d / -f 4 | cut -d . -f 1`
     fout="ptxgen/data/asm/"$f".sass"
     echo $fout
     cuobjdump --gpu-architecture sm_35 --dump-sass $p > $fout
done

echo "Gathering result from ptxgen...."
echo ".................................................................."

### put all sass result in one file ####
cat ptxgen/data/asm/*.sass > all.sass
### ignore  non instruction lines ####
ptxgen/extract.awk all.sass > all_inst.sass
### make it uniq ###
python ptxgen/uniq.py all_inst.sass > sm35.sass
BLOCK
echo "sm35.sass is generated as input to instruction encoding solver...."
echo ".................................................................."
############# Running opcode solver ####################
echo "Running opcode solvers............................................"
echo "It may take 10 miniutes ....................................."
echo ".................................................................."
time python opcode/opcode.py sm35.sass Kepler >opcode.txt
echo "Runing modifier solvers ..................."
python modifier/modifier.py sm35.sass Kepler >modifier.txt
echo "Runing operand solvers ...................."
time python operand/operand.py opcode.sass Kepler >operand.txt
