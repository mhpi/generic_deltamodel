#!/bin/sh

# Recursively remove all cache files (.pyc, __pycache__/) in the working dir.
# Usage: sh ../deltaModel/core/utils/clean_temp.sh 

find . | grep -E "(__pycache__|\.pyc$)" | xargs rm -rf
