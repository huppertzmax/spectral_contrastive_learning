#!/bin/bash
#SBATCH --job-name=train_job          
#SBATCH -p lrz-dgx-1-p100x8              
#SBATCH --gres=gpu:1                  
#SBATCH --time=600               
#SBATCH -o output_log_pretrain_spectral-contrastive-loss.out
#SBATCH -e error_log_pretrain_spectral-contrastive-loss.err   

enroot create --name ssl_container1 ./code/container/kernel-infonce.sqsh
enroot start -m code/ ssl_container1 << 'EOF'

pip install ipdb

cd ../code/spectral_contrastive_learning

python pretrain.py -c configs/spectral_resnet_mlp1000_norelu_cifar10_lr003_mu1.yaml

EOF
