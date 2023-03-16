if [ -z "$1" ]
    then
    echo "Require NGPU input; "
    exit
fi
loss="mse_sum"
NGPU=$1 ## 1 #8
num_node=2
mem=32
BS=10 
lr=2e-4
ENT="python train_dist.py --num_process_per_node $NGPU "
train_vae=False
cmt="lion"
ckpt="./lion_ckpt/unconditional/car/checkpoints/vae_only.pt"

$ENT \
    --config "./lion_ckpt/unconditional/car/cfg.yml" \
    latent_pts.pvd_mse_loss 1 \
    vis_latent_point 1 \
    num_val_samples 24 \
    ddpm.ema 1 \
    ddpm.use_bn False ddpm.use_gn True \
    ddpm.time_dim 64 \
    ddpm.beta_T 0.02 \
    sde.vae_checkpoint $ckpt \
    sde.learning_rate_dae $lr sde.learning_rate_min_dae $lr \
    trainer.epochs 18000 \
    sde.num_channels_dae 2048 \
    sde.dropout 0.3 \
    latent_pts.style_prior 'models.score_sde.resnet.PriorSEDrop' \
    sde.prior_model 'models.latent_points_ada_localprior.PVCNN2Prior' \
    sde.train_vae $train_vae \
    sde.embedding_scale 1.0 \
    viz.save_freq 1000 \
    viz.viz_freq -200 viz.log_freq -1 viz.val_freq -10000 \
    data.batch_size $BS \
    trainer.type 'trainers.train_2prior' \
    cmt $cmt 
