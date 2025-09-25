WANDB="+logging_params.wandb.project=polar-express" 
COMMON="+training_data=fineweb hydra.job.name=\"ns-steps\""
MUON="optimizer_params.name=muon"

for ns_steps in 3 4 5 6 7 ; do
for lr in 0.005; do
for polar_method in polarexpress; do
./submit_hydra.sh test_hydra $COMMON $MUON +optimizer_params.args.ns_steps=$ns_steps +optimizer_params.args.polar_method=$polar_method optimizer_params.args.lr=$lr optimizer_params.args.weight_decay=0.1 $WANDB
done
done
done