WANDB="+logging_params.wandb.project=polar-express" 
COMMON="+training_data=fineweb hydra.job.name=\"supression\""
MUON="optimizer_params.name=muon"

for lr in 0.005; do

for reverse in false true; do
for cutoff in 0.1 0.01 0.001 0.0001; do
./submit_hydra.sh test_hydra $COMMON $MUON +optimizer_params.args.ns_steps=0 +optimizer_params.args.polar_args.svd_cutoff=$cutoff +optimizer_params.args.polar_args.svd_reverse=$reverse +optimizer_params.args.polar_method="svd-exact" optimizer_params.args.lr=$lr optimizer_params.args.weight_decay=0.1 $WANDB
done
done

done

# on 1 H100, we expect each run to take 4.5 hours, but I'll give it 6 to be safe
