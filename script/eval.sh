python train_dist.py --skip_nll 1 --eval_generation --pretrained  $1 ddpm.model_var_type "fixedlarge" data.batch_size_test 3 ddpm.ema 1 num_val_samples 3 
