NP=2048
model="./lion_ckpt/unconditional/chair/checkpoints/model.pt"
python train_dist.py --eval_generation --pretrained  $model --skip_nll 0  \
    data.batch_size_test 32 ddpm.ema 1 trainer.type trainers.encode_interp_interp num_val_samples 20 trainer.seed 2 sde.ode_sample 1 \
    sde.beta_end 20.0 sde.embedding_scale 1000.0 \
    data.tr_max_sample_points ${NP} data.te_max_sample_points ${NP} shapelatent.decoder_num_points ${NP}

