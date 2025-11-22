WANDB="+logging_params.wandb.project=polar-express" 
COMMON="+training_data=fineweb hydra.job.name=\"ns-steps\""
MUON="optimizer_params.name=muon"

for lr in 0.005; do
for ns_steps in 2 3 4 5 6 7 10 20 30; do
./submit_hydra.sh test_hydra $COMMON $MUON +optimizer_params.args.ns_steps=$ns_steps +optimizer_params.args.polar_method="polarexpress" optimizer_params.args.lr=$lr optimizer_params.args.weight_decay=0.1 $WANDB
done

./submit_hydra.sh test_hydra $COMMON $MUON +optimizer_params.args.ns_steps=0 +optimizer_params.args.polar_method="svd-exact" optimizer_params.args.lr=$lr optimizer_params.args.weight_decay=0.1 $WANDB

done

# on 1 H100, we expect each run to take 4.5 hours, but I'll give it 6 to be safe
