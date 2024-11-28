#!/bin/bash
source /apps/profiles/modules_asax.sh.dyn
source /opt/asn/etc/asn-bash-profiles-special/modules.sh

module load cuda/11.7.0

nvcc -O3 -o hw5 hw5.cu 
nvcc -O3 -o hw5_pre_opt hw5_pre_opt.cu

nsys profile --stats=true ./hw5 5000 5000 /home/ualclsd0146/HW5_CS481/output
nsys profile --stats=true ./hw5 5000 5000 /home/ualclsd0146/HW5_CS481/output
nsys profile --stats=true ./hw5 5000 5000 /home/ualclsd0146/HW5_CS481/output

nsys profile --stats=true ./hw5_pre_opt 5000 5000 /home/ualclsd0146/HW5_CS481/output
nsys profile --stats=true ./hw5_pre_opt 5000 5000 /home/ualclsd0146/HW5_CS481/output
nsys profile --stats=true ./hw5_pre_opt 5000 5000 /home/ualclsd0146/HW5_CS481/output


nsys profile --stats=true ./hw5 10000 5000 /home/ualclsd0146/HW5_CS481/output
nsys profile --stats=true ./hw5 10000 5000 /home/ualclsd0146/HW5_CS481/output
nsys profile --stats=true ./hw5 10000 5000 /home/ualclsd0146/HW5_CS481/output

nsys profile --stats=true ./hw5_pre_opt 10000 5000 /home/ualclsd0146/HW5_CS481/output
nsys profile --stats=true ./hw5_pre_opt 10000 5000 /home/ualclsd0146/HW5_CS481/output
nsys profile --stats=true ./hw5_pre_opt 10000 5000 /home/ualclsd0146/HW5_CS481/output








