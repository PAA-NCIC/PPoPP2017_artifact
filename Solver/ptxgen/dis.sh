cubin=`ls cubin/*.cubin`

for p in $cubin
do
     f=`echo $p | cut -d . -f 1 |cut -d / -f 2`
     fout="asm/"$f".sass"
     echo $p
     #$cubin
     cuobjdump --gpu-architecture sm_21 --dump-sass $p > $fout
done
