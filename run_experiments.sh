#!/bin/bash

python siamese.py -t -n 30000 -s ff_30k_v2
python siamese.py -t -d -n 30000 -s ff_dist_30k_v2
python siamese.py -t -c -n 30000 -s cnn_30k_v2
python siamese.py -t -d -c -n 30000 -s cnn_d_30k_v2