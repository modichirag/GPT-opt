WANDB="+logging_params.wandb.project=polar-express"
COMMON="+training_data=fineweb"
MUON="optimizer_params.name=muon +optimizer_params.args.ns_steps=5"

for lr in 0.0005 0.001 0.005 0.01 0.02; do
./submit_hydra.sh test_hydra $COMMON optimizer_params.name=adamw optimizer_params.args.lr=$lr $WANDB
done

for lr in 0.005 0.01 0.05; do
for polar_method in Keller Jiacheng polarexpress; do
./submit_hydra.sh test_hydra $COMMON $MUON +optimizer_params.args.polar_method=$polar_method optimizer_params.args.lr=$lr $WANDB
done
done

