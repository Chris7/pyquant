#!/bin/bash
BASEDIR=$PWD
DATADIR="$BASEDIR/../tests/data"
mkdir -p $DATADIR
MSFFILE="$DATADIR/Chris_Ecoli_1-2-4-(01).msf"
MZMLFILE="$DATADIR/Chris_Ecoli_1-2-4.mzML"
if [ ! -e $MSFFILE ]; then
	echo "saving to $MSFFILE"
	wget -q -O $MSFFILE "ftp://ftp.pride.ebi.ac.uk/pride/data/archive/2016/03/PXD003327/Chris_Ecoli_1-2-4-(01).msf"
fi
if [ ! -e $MZMLFILE ]; then
	echo "saving to $MZMLFILE"
        wget -q -O $MZMLFILE "ftp://ftp.pride.ebi.ac.uk/pride/data/archive/2016/03/PXD003327/Chris_Ecoli_1-2-4.mzML"
fi
