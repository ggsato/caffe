#!/usr/bin/env sh

TOOLS=$CAFFE_ROOT/build/tools

$TOOLS/caffe train --solver=solver.prototxt 
