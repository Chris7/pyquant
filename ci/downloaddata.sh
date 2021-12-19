#!/bin/bash
BASEDIR=`dirname $0`
DATADIR="$BASEDIR/../tests/data"
mkdir -p $DATADIR
MSFFILE="$DATADIR/Chris_Ecoli_1-2-4-(01).msf"
MZMLFILE="$DATADIR/Chris_Ecoli_1-2-4.mzML"
if [ ! -e $MSFFILE ]; then
	echo "saving to $MSFFILE"
	wget -q -O $MSFFILE "https://ftp.ebi.ac.uk/pride-archive/2016/03/PXD003327/Chris_Ecoli_1-2-4-(01).msf"
fi
if [ ! -e $MZMLFILE ]; then
	echo "saving to $MZMLFILE"
  wget -q -O $MZMLFILE "https://ftp.ebi.ac.uk/pride-archive/2016/03/PXD003327/Chris_Ecoli_1-2-4.mzML"
fi
