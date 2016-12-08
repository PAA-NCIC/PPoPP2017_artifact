arch=sm_35
mkdir -p ptxgen/data/ptx
echo "generating sass files in ptxgen/data/ptx directory............... "
echo ".................................................................."
ptxgen/ptxgen sm_35

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
     echo $fout
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
