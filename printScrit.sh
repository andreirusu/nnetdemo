#!/bin/bash
for i in {0..20} 
do 
    j=$(echo 2 k $i 10 / p | dc)
    ./nnetplot.lua -network demo/mlp.net -b $j
done 
