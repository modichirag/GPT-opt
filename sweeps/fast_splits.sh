WANDB="+logging_params.wandb.project=polar-express"
COMMON="+training_data=fineweb hydra.job.name=\"fast_splits\""
MUON="optimizer_params.name=muon +optimizer_params.args.ns_steps=5"

for lr in 0.003 0.005 0.01; do
for split_heads in True False; do
for polar_method in fast_polarexpress polarexpress; do
./submit_hydra.sh test_hydra $COMMON $MUON +optimizer_params.args.polar_method=$polar_method optimizer_params.args.lr=$lr +optimizer_params.args.split_heads=$split_heads $WANDB
done
done
done

