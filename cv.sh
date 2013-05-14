#!/bin/bash

echo OPTIONS:$*

DATASET_PATH=~/bblch/data/kfolds

sum=0
k=4

rm -Rf res.txt FOLD*

for i in $(seq 1 $k)
do  
    echo FOLD$i 
    torch nnetdemo_classification.lua -dataset $DATASET_PATH/$i/fold -save FOLD$i $*  
    acc=$(cat  FOLD$i/test.accuracy.txt | tail -n1)
    sum=`echo "$sum + $acc" | bc`
done

sum=`echo "$sum / $k * 10000" | bc -l`
echo $sum > res.txt

cat res.txt



