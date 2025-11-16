WANDB="+logging_params.wandb.project=polar-express"
COMMON="+training_data=fineweb10B gpt_model=gpt-large hydra.job.name=\"10b_data\""
MUON="optimizer_params.name=muon +optimizer_params.args.ns_steps=5"

for wd in 0.1; do

for lr in 0.01 0.02 0.05; do
for polar_method in Jiacheng polarexpress; do
./submit_hydra.sh test_hydra $COMMON $MUON +optimizer_params.args.polar_method=$polar_method optimizer_params.args.lr=$lr optimizer_params.args.weight_decay=$wd $WANDB
done
done

done