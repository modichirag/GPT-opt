import hashlib
import json

def hash_config(optimizer_config, training_params, gpt_model):
    """
    Generate a hash from the relevant fields of the current optimizer configuration,
    training parameters, and GPT model configuration.

    Parameters
    ----------
    optimizer_config : dict
        The configuration dictionary for the current optimizer.
    training_params : dict
        The training parameters dictionary.
    gpt_model : dict
        The GPT model configuration dictionary.

    Returns
    -------
    str
        A compressed hash string.
    """
    # Combine relevant fields
    relevant_fields = {
        "optimizer_config": optimizer_config,
        "training_params": training_params,
        "gpt_model": gpt_model
    }
    # Convert to a JSON string and hash it
    config_str = json.dumps(relevant_fields, sort_keys=True)
    return hashlib.md5(config_str.encode()).hexdigest()
