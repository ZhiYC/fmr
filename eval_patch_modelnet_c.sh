#!/bin/bash
set -x
if [ ! -d $1 ];then
    echo "Error: $1 is not a directory"
    exit 1
fi

for file in `ls $1`
do  
    if test -d $1$file
    then
        echo "processing: $1$file"
        mv $1$file ~/tmp/
        rm -rf ./data/ModelNet40
        ln -s ~/tmp/$file/ModelNet40 ./data/ModelNet40
        python evaluate.py -data modelnet --perturbations ./data/pert_030.csv --pretrained ./result/fmr_my_modelnet.pth > $file".log"
        mv ~/tmp/$file $1$file"/"
    fi
done

#EOF