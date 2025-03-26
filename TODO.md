1. Change the way models are loaded. Don't use pre-trained models. Use tokenizer same as model. Have only one way to load a model (name or numbers of layers)
2. change the way data is loaded, to be streaming instead of loading everything
3. Use some mixed precision in training.
4. Modifications for scaling up: gradient accumulation? Multiple workers for loading data? Checkpoint the model in case?

