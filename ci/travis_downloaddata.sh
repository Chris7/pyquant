#!/bin/bash
BASEDIR=$PWD
DATADIR="$BASEDIR/../pyquant/tests/data"
mkdir -p $DATADIR
MSFFILE="$DATADIR/Chris_Ecoli_1-2-4-(01).msf"
MZMLFILE="$DATADIR/Chris_Ecoli_1-2-4.mzML"
if [ ! -e $MSFFILE ]; then
	wget -O $MSFFILE "ftp://ftp.pride.ebi.ac.uk/pride/data/archive/2016/03/PXD003327/Chris_Ecoli_1-2-4-(01).msf"
fi
if [ ! -e $MZMLFILE ]; then
        wget -O $MZMLFILE "ftp://ftp.pride.ebi.ac.uk/pride/data/archive/2016/03/PXD003327/Chris_Ecoli_1-2-4.mzML"
fi
