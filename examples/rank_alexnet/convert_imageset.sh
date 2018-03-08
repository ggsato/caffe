#! /usr/bin/env sh

TOOLS=$CAFFE_ROOT/build/tools

SIZE=$1
ROOT_FOLDER=$2
IMG_LIST=$3
OUTPUT=$4

$TOOLS/convert_imageset -resize_height $1 -resize_width $SIZE $ROOT_FOLDER $IMG_LIST $OUTPUT