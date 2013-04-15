#!/bin/bash
cmd="./nnetplot.lua -network demo/mlp.net -size 1000 -c"
for i in {1..20}
do
    j=$(echo 2 k $i 10 / p | dc)
    echo  $cmd $j
    
    $cmd $j > /dev/null
done

