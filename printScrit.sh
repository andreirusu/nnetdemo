#!/bin/bash
cmd="./nnetplot.lua -network demo/mlp.net -d" 
for i in {1..20} 
do 
    j=$(echo 2 k $i 100 / p | dc)
    echo  $cmd $j 
    
    $cmd $j > /dev/null
done 
