#!/bin/zsh

for filename in *.nc; do 
    [ -f "$filename" ] || continue
    mv "$filename" "${filename//($1)/}"

done
