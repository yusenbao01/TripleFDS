import torch
import torch.nn as nn
import os


def load_checkpoint_adaptive(
    model,
    checkpoint_path,
    verbose=True,
    map_location="cpu",
    strict=False,
    ignore_modules=None,
    load_modules=None,
    trainer=None,
):
    """
    Adaptively loads a checkpoint, only loading parameters that exist in the model.
    New parameters in the model will use their default initialization.

    Args:
        model: The model to load parameters into.
        checkpoint_path: Path to the checkpoint file.
        verbose: Whether to print detailed information.
        map_location: Device mapping parameter for torch.load.
        strict: Whether to raise an exception if no parameters are loaded.
        ignore_modules: A list/set of module names to ignore. These modules will use default initialization.
        load_modules: A list/set of module names to exclusively load.

    Returns:
        model: The model with loaded weights.
    """
    if trainer is not None:
        verbose = verbose and trainer.is_global_zero

    # Ensure ignore_modules and load_modules are sets for efficient lookup
    ignore_modules = set(ignore_modules) if ignore_modules else set()
    load_modules = set(load_modules) if load_modules else None

    if not checkpoint_path or not os.path.exists(checkpoint_path):
        if verbose:
            print(f"Checkpoint file not found: {checkpoint_path}")
        return model

    if verbose:
        print(f"Loading checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)

    # Adapt to different checkpoint formats
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get("state_dict") or checkpoint.get("model") or checkpoint
    else:
        state_dict = checkpoint

    current_state_dict = model.state_dict()

    # Remove the 'module.' prefix if it exists
    clean_state_dict = {(k[len("module.") :] if k.startswith("module.") else k): v for k, v in state_dict.items()}

    new_state_dict = {}
    missing_keys, unexpected_keys, ignored_by_rule_keys, ignored_by_load_keys, shape_mismatch_keys = (
        [],
        list(clean_state_dict.keys()),
        [],
        [],
        [],
    )

    for k, v in current_state_dict.items():
        # Check if the key is in ignore_modules
        if any(k.startswith(module_name) for module_name in ignore_modules):
            ignored_by_rule_keys.append(k)
            continue

        # If load_modules is provided, check if the key is in it
        if load_modules and not any(k.startswith(module_name) for module_name in load_modules):
            ignored_by_load_keys.append(k)
            continue

        if k in clean_state_dict:
            if v.shape == clean_state_dict[k].shape:
                new_state_dict[k] = clean_state_dict[k]
            else:
                shape_mismatch_keys.append(f"{k} (model: {v.shape}, ckpt: {clean_state_dict[k].shape})")
            unexpected_keys.remove(k)
        else:
            missing_keys.append(k)

    if new_state_dict:
        model.load_state_dict(new_state_dict, strict=False)
        if verbose:
            print(f"\nSuccessfully loaded {len(new_state_dict)} parameters.")
            if ignored_by_rule_keys:
                print(f"Ignored {len(ignored_by_rule_keys)} parameters based on 'ignore_modules' rules.")
            if load_modules and ignored_by_load_keys:
                print(f"Ignored {len(ignored_by_load_keys)} parameters based on 'load_modules' rules.")
            if missing_keys:
                print(f"Parameters missing in checkpoint ({len(missing_keys)}): {', '.join(missing_keys[:5])}...")
            if unexpected_keys:
                print(f"Parameters in checkpoint but not in model ({len(unexpected_keys)}): {', '.join(unexpected_keys[:5])}...")
            if shape_mismatch_keys:
                print(f"Parameters with shape mismatch ({len(shape_mismatch_keys)}): {', '.join(shape_mismatch_keys[:5])}...")

    elif strict:
        raise RuntimeError("No loadable parameters found.")
    elif verbose:
        print("Warning: No loadable parameters were found.")
        print("=" * 80)
        print("To help with troubleshooting, here are some sample key names from the model and checkpoint:")

        model_keys = list(current_state_dict.keys())
        print(f"\nTop 5 key names in the model (total {len(model_keys)}):")
        for i in range(min(5, len(model_keys))):
            print(f"  - {model_keys[i]}")

        ckpt_keys = list(clean_state_dict.keys())
        print(f"\nTop 5 key names in the checkpoint (total {len(ckpt_keys)}):")
        for i in range(min(5, len(ckpt_keys))):
            print(f"  - {ckpt_keys[i]}")

        print("\nPlease check:")
        print("1. If using `load_modules`, does its prefix match any of the 'key names in the model'?")
        print("2. Do the 'key names in the model' and 'key names in the checkpoint' share a common suffix but have different prefixes?")
        print("3. Is the checkpoint file in the expected format?")
        print("=" * 80)

    return model
