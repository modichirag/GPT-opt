WANDB="+logging_params.wandb.project=polar-express"
COMMON="+training_data=fineweb10B gpt_model=gpt-small hydra.job.name=\"10b_data_small\""
MUON="optimizer_params.name=muon +optimizer_params.args.ns_steps=5"

for wd in 0.0; do

for lr in 0.002 0.005 0.01 0.02; do
for polar_method in Keller Jiacheng polarexpress; do
./submit_hydra.sh test_hydra $COMMON $MUON +optimizer_params.args.polar_method=$polar_method optimizer_params.args.lr=$lr optimizer_params.args.weight_decay=$wd $WANDB
done
done

done

# on 1 H100, we expect each run to take 12.5 hours, but I'll give it 16 to be safe
