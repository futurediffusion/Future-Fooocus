from __future__ import annotations
import gradio as gr
import logging
import os
import re

import functools
import network

import torch
from safetensors.torch import load_file
from typing import Union

from modules import sd_models, config
from ldm_patched.modules.lora import model_lora_keys_clip, model_lora_keys_unet, load_lora


def load_lora_for_models(model, clip, lora, strength_model, strength_clip, filename='default'):
    model_flag = type(model.model).__name__ if model is not None else 'default'

    unet_keys = model_lora_keys_unet(model.model) if model is not None else {}
    clip_keys = model_lora_keys_clip(clip.cond_stage_model) if clip is not None else {}

    # ldm_patched.modules.lora.load_lora() returns only the dictionary of
    # patches to apply. Previous versions of this function returned both the
    # patches and the remaining keys, which resulted in a ``ValueError`` when
    # trying to unpack two values here.  Instead of expecting two values, load
    # the LoRA weights separately for UNet and CLIP.

    # load_lora() will print warnings for any unused keys itself, so there is no
    # need to keep track of ``lora_unmatch`` here.
    lora_unet = load_lora(lora, unet_keys)
    lora_clip = load_lora(lora, clip_keys)

    # The legacy implementation reported unmatched keys when the LoRA file
    # contained parameters that did not correspond to either the UNet or the
    # CLIP model. ``load_lora`` already logs such keys, so explicit handling is
    # no longer required here.

    new_model = model.clone() if model is not None else None
    new_clip = clip.clone() if clip is not None else None

    if new_model is not None and len(lora_unet) > 0:
        loaded_keys = new_model.add_patches(lora_unet, strength_model)
        skipped_keys = [item for item in lora_unet if item not in loaded_keys]
        if len(skipped_keys) > 12:
            print(f'[LORA] Mismatch {filename} for {model_flag}-UNet with {len(skipped_keys)} keys mismatched in {len(loaded_keys)} keys')
        else:
            print(f'[LORA] Loaded {filename} for {model_flag}-UNet with {len(loaded_keys)} keys at weight {strength_model} (skipped {len(skipped_keys)} keys)')
            model = new_model

    if new_clip is not None and len(lora_clip) > 0:
        loaded_keys = new_clip.add_patches(lora_clip, strength_clip)
        skipped_keys = [item for item in lora_clip if item not in loaded_keys]
        if len(skipped_keys) > 12:
            print(f'[LORA] Mismatch {filename} for {model_flag}-CLIP with {len(skipped_keys)} keys mismatched in {len(loaded_keys)} keys')
        else:
            print(f'[LORA] Loaded {filename} for {model_flag}-CLIP with {len(loaded_keys)} keys at weight {strength_clip} (skipped {len(skipped_keys)} keys)')
            clip = new_clip

    return model, clip


@functools.lru_cache(maxsize=5)
def load_lora_state_dict(filename):
    ext = os.path.splitext(filename)[-1].lower()
    if ext == ".safetensors":
        return load_file(filename, device="cpu")
    else:
        # PyTorch >=2.6 defaults to ``weights_only=True``. Some older LoRA files
        # fail to load in this mode.  Attempt ``weights_only=True`` first for
        # security, then fall back to the legacy behaviour if it fails.
        try:
            return torch.load(filename, map_location="cpu", weights_only=True)
        except Exception:
            print(f"[LoRA loader fallback] {filename} failed with weights_only=True, retrying with weights_only=False")
            return torch.load(filename, map_location="cpu", weights_only=False)


def load_network(name, network_on_disk):
    net = network.Network(name, network_on_disk)
    net.mtime = os.path.getmtime(network_on_disk.filename)

    return net


def load_networks(names, te_multipliers=None, unet_multipliers=None, dyn_dims=None):
    current_sd = sd_models.model_data.get_sd_model()
    if current_sd is None:
        return

    loaded_networks.clear()

    unavailable_networks = []
    for name in names:
        if name.lower() in forbidden_network_aliases and available_networks.get(name) is None:
            unavailable_networks.append(name)
        elif available_network_aliases.get(name) is None:
            unavailable_networks.append(name)

    if unavailable_networks:
        update_available_networks_by_names(unavailable_networks)

    networks_on_disk = [available_networks.get(name, None) if name.lower() in forbidden_network_aliases else available_network_aliases.get(name, None) for name in names]
    if any(x is None for x in networks_on_disk):
        list_available_networks()
        networks_on_disk = [available_networks.get(name, None) if name.lower() in forbidden_network_aliases else available_network_aliases.get(name, None) for name in names]

    for i, (network_on_disk, name) in enumerate(zip(networks_on_disk, names)):
        try:
            net = load_network(name, network_on_disk)
        except Exception as e:
            print(f"Error loading network {network_on_disk.filename}: {e}")
            continue
        net.mentioned_name = name
        network_on_disk.read_hash()
        loaded_networks.append(net)

    compiled_lora_targets = []
    for a, b, c in zip(networks_on_disk, unet_multipliers, te_multipliers):
        compiled_lora_targets.append([a.filename, b, c])

    compiled_lora_targets_hash = str(compiled_lora_targets)

    if current_sd.current_lora_hash == compiled_lora_targets_hash:
        return

    current_sd.current_lora_hash = compiled_lora_targets_hash
    current_sd.forge_objects.unet = current_sd.forge_objects_original.unet
    current_sd.forge_objects.clip = current_sd.forge_objects_original.clip

    for filename, strength_model, strength_clip in compiled_lora_targets:
        lora_sd = load_lora_state_dict(filename)
        current_sd.forge_objects.unet, current_sd.forge_objects.clip = load_lora_for_models(
            current_sd.forge_objects.unet, current_sd.forge_objects.clip, lora_sd, strength_model, strength_clip,
            filename=filename)

    current_sd.forge_objects_after_applying_lora = current_sd.forge_objects.shallow_copy()
    return


def process_network_files(names: list[str] | None = None):
    candidates = []
    for folder in config.paths_loras:
        if not os.path.isdir(folder):
            continue
        for root_dir, _dirs, files in os.walk(folder):
            for file in files:
                if os.path.splitext(file)[1].lower() in [".pt", ".ckpt", ".safetensors"]:
                    candidates.append(os.path.join(root_dir, file))
    for filename in candidates:
        if os.path.isdir(filename):
            continue
        name = os.path.splitext(os.path.basename(filename))[0]
        # if names is provided, only load networks with names in the list
        if names and name not in names:
            continue
        try:
            entry = network.NetworkOnDisk(name, filename)
        except OSError as e:  # should catch FileNotFoundError and PermissionError etc.
            print(f"Failed to load network {name} from {filename}: {e}")
            continue

        available_networks[name] = entry

        if entry.alias in available_network_aliases:
            forbidden_network_aliases[entry.alias.lower()] = 1

        available_network_aliases[name] = entry
        available_network_aliases[entry.alias] = entry


def update_available_networks_by_names(names: list[str]):
    process_network_files(names)


def list_available_networks():
    available_networks.clear()
    available_network_aliases.clear()
    forbidden_network_aliases.clear()
    available_network_hash_lookup.clear()
    forbidden_network_aliases.update({"none": 1, "Addams": 1})

    for folder in config.paths_loras:
        os.makedirs(folder, exist_ok=True)

    process_network_files()


re_network_name = re.compile(r"(.*)\s*\([0-9a-fA-F]+\)")


def infotext_pasted(infotext, params):
    added = []

    for k in params:
        if not k.startswith("AddNet Model "):
            continue

        num = k[13:]

        if params.get("AddNet Module " + num) != "LoRA":
            continue

        name = params.get("AddNet Model " + num)
        if name is None:
            continue

        m = re_network_name.match(name)
        if m:
            name = m.group(1)

        multiplier = params.get("AddNet Weight A " + num, "1.0")

        added.append(f"<lora:{name}:{multiplier}>")

    if added:
        params["Prompt"] += "\n" + "".join(added)


extra_network_lora = None

available_networks = {}
available_network_aliases = {}
loaded_networks = []
loaded_bundle_embeddings = {}
networks_in_memory = {}
available_network_hash_lookup = {}
forbidden_network_aliases = {}

list_available_networks()
