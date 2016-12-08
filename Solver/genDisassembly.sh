#!/bin/bash
echo "Please input architecture parameter argv[1]:sm_35|sm_52"
if [ "$#" -lt 1 ]
then
    echo "Error parameters, argv[1]: sm_35|sm_52"
    exit -1
else
    if [ $1 != "sm_35" -a $1 != "sm_52" ]
    then
        echo "Error parameter ..."
        exit -2
    fi
fi

arch=$1 #can be sm_35 or sm_52
prefix="ptxgen/data_"$arch"/"

ptx=$prefix"ptx/*.ptx"

mkdir -p $prefix"ptx"
ptxgen/ptxgen $arch $prefix"ptx"
mkdir -p $prefix"cubin"
echo "compile .ptx file to cubin files in ptxgen/data/cubin directory..."
echo "It may take serveral miniutes..."
echo ".................................................................."
for p in $ptx
do
     f=`echo $p | cut -d / -f 4 |cut -d . -f 1` 
     fout=$prefix"cubin/"$f".cubin"
     echo $fout
     ptxas -arch $arch -m 64 $p -o $fout  > /dev/null 2>&1
done

cubin=$prefix"cubin/*.cubin"
########## disassembly it using cubin ########
echo "disassble .cubin file to sass files in ptxgen/data/asm directory..."
echo "It may take serveral miniutes..."
echo ".................................................................."
mkdir -p $prefix"asm"
for p in $cubin
do
     f=`echo $p | cut -d / -f 4 | cut -d . -f 1`
     fout=$prefix"asm/"$f".sass"
     echo $fout
     cuobjdump --gpu-architecture $arch --dump-sass $p > $fout
done
asm=$prefix"asm/*.sass"

echo "Gathering result from ptxgen...."
echo ".................................................................."
### put all sass result in one file ####

#cat $prefix"asm/*.sass" > all.sass
if [ -f all.sass ]
then
    rm all.sass
else
    touch all.sass
fi

for f in $asm
do
    cat $f >> all.sass
done
### ignore  non instruction lines ####
ptxgen/extract.awk all.sass > all_inst.sass
### make instruction uniq ###
python ptxgen/uniq.py all_inst.sass > $arch".sass"

rm all.sass all_inst.sass
