ptx=`ls ptx/*.ptx`

for p in $ptx
do
     f=`echo $p | cut -d . -f 1 |cut -d / -f 2`
     fout="cubin/"$f".cubin"
     echo $fout
     ptxas -arch sm_35 -m 64 $p -o $fout
done
