#!/bin/bash
for filename in ./*.jpg; do
    convert $filename -colorspace Gray "$filename""_bw.jpg"
done
