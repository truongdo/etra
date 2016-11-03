#! /bin/bash
#
# path.sh
# Copyright (C) 2016 truong-d <truong-d@truongd-ThinkPad-X1-Carbon-3rd>
#
# Distributed under terms of the MIT license.
#


ETRA=`pwd`/../../
export PATH=${ETRA}/bin:`pwd`/steps:`pwd`/utils/:${PATH}
cmd=run.pl
