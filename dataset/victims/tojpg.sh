#!/bin/bash
i=9
for file in `ls *.pgm`;do mv $file ${file%.pgm*}$i.jpg;done