import os
import json
import requests
import numpy as np
import torch
from tqdm import tqdm
from src.config.config import CONFIG, logger


def download_file(url, destination):
    """Downloads a file from a URL to a local destination path.

    Args:
        url (str): The URL of the file to download.
        destination (str): The local file path where the file should be saved.

    Raises:
        requests.exceptions.RequestException: If the download fails.
    """
    response = requests.get(url, stream=True, verify=True, timeout=120)
    response.raise_for_status()

    file_size = int(response.headers.get("content-length", 0))

    if os.path.exists(destination):
        file_size_local = os.path.getsize(destination)
        if file_size > 0 and file_size == file_size_local:
            logger.info(f"File already exists and is up-to-date: {destination}")
            return
        if file_size == 0 and file_size_local > 0:
            logger.info(f"File already exists (unknown remote size): {destination}")
            return

    block_size = 1024
    progress_bar_description = url.split("/")[-1]
    temp_destination = f"{destination}.partial"
    bytes_written = 0
    with tqdm(total=file_size or None, unit="iB", unit_scale=True, desc=progress_bar_description) as progress_bar:
        with open(temp_destination, "wb") as file:
            for chunk in response.iter_content(block_size):
                if not chunk:
                    continue
                progress_bar.update(len(chunk))
                file.write(chunk)
                bytes_written += len(chunk)

    if file_size > 0 and bytes_written != file_size:
        os.remove(temp_destination)
        raise IOError(
            f"Incomplete download for {destination}: expected {file_size} bytes, "
            f"received {bytes_written}"
        )

    os.replace(temp_destination, destination)


def load_gpt2_params_from_tf_ckpt(ckpt_path, settings):
    """Loads GPT-2 parameters from a TensorFlow checkpoint into a nested dictionary.

    Args:
        ckpt_path (str): The path to the TensorFlow checkpoint directory.
        settings (dict): The hyperparameters dictionary containing 'n_layer'.

    Returns:
        dict: A nested dictionary containing the parsed model weights.
    """
    import tensorflow as tf
    
    params = {"blocks": [{} for _ in range(settings["n_layer"])]}

    for name, _ in tf.train.list_variables(ckpt_path):
        variable_array = np.squeeze(tf.train.load_variable(ckpt_path, name))
        variable_name_parts = name.split("/")[1:]

        target_dict = params
        if variable_name_parts[0].startswith("h"):
            layer_number = int(variable_name_parts[0][1:])
            target_dict = params["blocks"][layer_number]

        for key in variable_name_parts[1:-1]:
            target_dict = target_dict.setdefault(key, {})

        last_key = variable_name_parts[-1]
        target_dict[last_key] = variable_array

    return params


def download_and_load_gpt2(model_size, models_dir):
    """Downloads OpenAI's pre-trained GPT-2 model weights and configuration.

    Args:
        model_size (str): The size of the GPT-2 model (e.g., '124M', '355M').
        models_dir (str): The base directory where the model files should be saved.

    Returns:
        tuple: A tuple containing the parsed settings dictionary and the model weights dictionary.

    Raises:
        ValueError: If the requested model_size is not one of the allowed sizes.
    """
    import tensorflow as tf
    
    allowed_sizes = CONFIG["weights"]["gpt2_checkpoint_sizes"]
    if model_size not in allowed_sizes:
        raise ValueError(f"model_size must be one of {allowed_sizes}")

    model_dir = os.path.join(models_dir, model_size)
    base_url = "https://openaipublic.blob.core.windows.net/gpt-2/models"
    filenames = [
        "checkpoint", "encoder.json", "hparams.json",
        "model.ckpt.data-00000-of-00001", "model.ckpt.index",
        "model.ckpt.meta", "vocab.bpe"
    ]

    os.makedirs(model_dir, exist_ok=True)
    for filename in filenames:
        file_url = os.path.join(base_url, model_size, filename)
        file_path = os.path.join(model_dir, filename)
        download_file(file_url, file_path)

    tf_ckpt_path = tf.train.latest_checkpoint(model_dir)
    hparams_path = os.path.join(model_dir, "hparams.json")
    with open(hparams_path, encoding="utf-8") as hparams_file:
        settings = json.load(hparams_file)
    params = load_gpt2_params_from_tf_ckpt(tf_ckpt_path, settings)

    return settings, params


def assign(left, right):
    """Assigns a numpy array to a PyTorch parameter with shape validation.

    Args:
        left (torch.nn.Parameter): The target PyTorch parameter.
        right (numpy.ndarray): The source numpy array containing the weights.

    Returns:
        torch.nn.Parameter: A new PyTorch parameter initialized with the source weights.

    Raises:
        ValueError: If the shapes of the target parameter and source array do not match.
    """
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
    param = torch.nn.Parameter(torch.tensor(right))
    param.requires_grad = left.requires_grad
    return param


def load_weights_into_gpt(gpt, params):
    """Loads pre-trained GPT-2 weights from a nested dictionary into a GPTModel instance.

    Args:
        gpt (GPTModel): The PyTorch GPT-2 model instance to populate.
        params (dict): The nested dictionary of weights parsed from the TensorFlow checkpoint.
    """
    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params['wpe'])
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params['wte'])

    for b in range(len(params["blocks"])):
        # OpenAI's TensorFlow implementation stores the Query, Key, and Value weights as a single 
        # concatenated tensor. We must split it into three separate matrices for our PyTorch architecture.
        q_w, k_w, v_w = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1)
        
        # We transpose the weights (.T) because PyTorch Linear layers expect weights 
        # in the shape (out_features, in_features), whereas TensorFlow stores them transposed.
        gpt.trf_blocks[b].att.W_query.weight = assign(
            gpt.trf_blocks[b].att.W_query.weight, q_w.T)
        gpt.trf_blocks[b].att.W_key.weight = assign(
            gpt.trf_blocks[b].att.W_key.weight, k_w.T)
        gpt.trf_blocks[b].att.W_value.weight = assign(
            gpt.trf_blocks[b].att.W_value.weight, v_w.T)

        q_b, k_b, v_b = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.bias = assign(
            gpt.trf_blocks[b].att.W_query.bias, q_b)
        gpt.trf_blocks[b].att.W_key.bias = assign(
            gpt.trf_blocks[b].att.W_key.bias, k_b)
        gpt.trf_blocks[b].att.W_value.bias = assign(
            gpt.trf_blocks[b].att.W_value.bias, v_b)

        gpt.trf_blocks[b].att.out_proj.weight = assign(
            gpt.trf_blocks[b].att.out_proj.weight,
            params["blocks"][b]["attn"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].att.out_proj.bias = assign(
            gpt.trf_blocks[b].att.out_proj.bias,
            params["blocks"][b]["attn"]["c_proj"]["b"])

        gpt.trf_blocks[b].ff.layers[0].weight = assign(
            gpt.trf_blocks[b].ff.layers[0].weight,
            params["blocks"][b]["mlp"]["c_fc"]["w"].T)
        gpt.trf_blocks[b].ff.layers[0].bias = assign(
            gpt.trf_blocks[b].ff.layers[0].bias,
            params["blocks"][b]["mlp"]["c_fc"]["b"])
        gpt.trf_blocks[b].ff.layers[2].weight = assign(
            gpt.trf_blocks[b].ff.layers[2].weight,
            params["blocks"][b]["mlp"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].ff.layers[2].bias = assign(
            gpt.trf_blocks[b].ff.layers[2].bias,
            params["blocks"][b]["mlp"]["c_proj"]["b"])

        gpt.trf_blocks[b].norm1.scale = assign(
            gpt.trf_blocks[b].norm1.scale,
            params["blocks"][b]["ln_1"]["g"])
        gpt.trf_blocks[b].norm1.shift = assign(
            gpt.trf_blocks[b].norm1.shift,
            params["blocks"][b]["ln_1"]["b"])
        gpt.trf_blocks[b].norm2.scale = assign(
            gpt.trf_blocks[b].norm2.scale,
            params["blocks"][b]["ln_2"]["g"])
        gpt.trf_blocks[b].norm2.shift = assign(
            gpt.trf_blocks[b].norm2.shift,
            params["blocks"][b]["ln_2"]["b"])

    gpt.final_norm.scale = assign(gpt.final_norm.scale, params["g"])
    gpt.final_norm.shift = assign(gpt.final_norm.shift, params["b"])
    
    # Weight Tying: GPT-2 uses the exact same weight matrix for both the input token embeddings 
    # and the final output classification head to save memory and improve representations.
    gpt.out_head.weight = assign(gpt.out_head.weight, params["wte"])
