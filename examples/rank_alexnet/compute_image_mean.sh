#! /usr/bin/env sh

TOOLS=$CAFFE_ROOT/build/tools

INPUT_DB=$1
OUTPUT=$2

$TOOLS/compute_image_mean $INPUT_DB $OUTPUT