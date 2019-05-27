jbsub  -err err/$1.txt -out out/$1.txt -cores 2x1+1 -require k80 -mem $3 -q x86_$2 /dccstor/cssblr/amrita/miniconda/bin/python -W ignore load_SSRP.py parameters/noisy/$1.json 
