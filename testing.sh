#!/bin/bash
source /apps/profiles/modules_asax.sh.dyn
source /opt/asn/etc/asn-bash-profiles-special/modules.sh

module load cuda/11.7.0

nvcc -O3 -o hw5 hw5.cu 

nsys profile --stats=true ./hw5 80 100 /home/ualclsd0146/HW5_CS481/output
nsys profile --stats=true ./hw5 5000 5000 /home/ualclsd0146/HW5_CS481/output





