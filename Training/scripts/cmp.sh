#!/bin/bash
# cmp.sh <ref> <new>  -> identical count, total change, max ratio
paste -d' ' $1 $2 | awk '{n++; eq+=($1==$2); s0+=$1; s1+=$2; r=$2/$1; if(r>mx)mx=r; l+=($2<$1)} END{printf "identical %d/%d  nodes %+.1f%%  fewer in %d  max ratio %.2f\n",eq,n,100*(s1/s0-1),l,mx}'
