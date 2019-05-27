jbsub -cores 1+1 -require k80 -err err/$2_$1.txt -out out/$2_$1.txt -mem 180g -q x86_24h /dccstor/cssblr/amrita/miniconda/bin/python train_SSRP.py parameters/parameters_$1.json $2
