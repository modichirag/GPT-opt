from run_parallel import run_parallel


family_idx = 40
family_name = "truncated"


base_config = {
    "optimizer_params": [{
        "name": "nesgd",
        "lr": [0.001],
        "weight_decay": 0.01,
        "lr_schedule": "constant-linear",
        "warm_up_fraction": 0.4,
        "spectral_scale": 16.0,
    }],
    "training_params": {
        "tokens_processed": 524288, # 2^19
        "val_tokens_processed": 8388608, #2^23
        "batch_size": 64,
        "num_epochs": 1,
        "context_length": 1024,
        "gradnorm": 1.0,
        "tensorcore_precision": "high",   #Can be highest, high, or medium
        "autocast": True,
        "mixed_precision": "bfloat16",
        "compile": True,
    },
    "logging_params": {
        "val_tokens_processed": 8388608,
        "log_step": 256,
        "val_step": 256,
        "save_ckpt_step": 512,
        "load_ckpt_step": 0,
        "keep_last": 2,
        "ckpt_dir": "",
    },
    "gpt_model": {
        "n_embd": 768,
        "n_layer": 12,
        "n_head": 12,
        "vocab_size": 50257,
        "flash_attention": True,
    },
    "dataset": {
        "name": "fineweb1B"
    },
}

# Generate configs for this experiment.
lrs = [0.00001, 0.0001, 0.001, 0.01, 0.1]
loss_lbs = [0.0, 1.6, 2.4, 2.8, 3.2]
alg_settings = {
    "muon-momo": {"spectral_scale": 16},
    "muonmax-momo": {"spectral_scale": 4},
    "scion-momo": {"spectral_scale": 4},
    "polargrad-momo": {"spectral_scale": 64},
}
alg_to_optimizer = {
    "muon-momo": "nesgd-adam_infty-lmo-momo",
    "muonmax-momo": "nesgd-adam_2-hybrid_prod-momo",
    "scion-momo": "nesgd-lmo-momo",
    "polargrad-momo": "nesgd-adam_2-l2_prod_norm-momo",
}

experiment_configs = {}
for alg, settings in alg_settings.items():
    optimizer_name = alg_to_optimizer[alg]
    for loss_lb in loss_lbs:
        opt_settings = {
            "name": optimizer_name,
            "lr": list(lrs),
            "truncate_loss": loss_lb,
        }
        opt_settings.update(alg_settings[alg])
        current_config = dict(base_config)
        current_config["optimizer_params"][0].update(dict(opt_settings))
        run_name = f"{family_idx}_{family_name}_{alg}_{loss_lb}"
        experiment_configs[run_name] = dict(current_config)

# Launch runs in parallel.
run_parallel(experiment_configs)
