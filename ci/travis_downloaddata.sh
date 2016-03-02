#!/bin/bash
BASEDIR=$PWD
DATADIR="$BASEDIR/../pyquant/tests/data"
if [ ! -e "$DATADIR/Chris_Ecoli_1-2-4-(01).msf" ]; then
	wget "ftp://ftp.pride.ebi.ac.uk/pride/data/archive/2016/03/PXD003327/Chris_Ecoli_1-2-4-(01).msf"
fi
if [ ! -e "$DATADIR/Chris_Ecoli_1-2-4.mzML" ]; then
        wget "ftp://ftp.pride.ebi.ac.uk/pride/data/archive/2016/03/PXD003327/Chris_Ecoli_1-2-4.mzML"
fi
