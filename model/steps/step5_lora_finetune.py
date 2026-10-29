"""
step5_lora_finetune
See model/steps/README.md and docs/plan.md for what this step does and why.

One change from the frozen backbone: LoRA adapters on the attention query/value
projections of the last `lora.last_n_blocks` blocks are trained, with a linear
head, on random native-resolution 224 px crops of the training images (with
step 3's augmentation if lora.augment). Everything else stays frozen; full
fine-tuning overfits to the training generators (docs/plan.md, Pitfalls).

After training, the adapters are merged and only the weight CHANGES are saved
(checkpoint key "weight_deltas", ~5-20 MB). Then, exactly as in steps 2-3, the
recipe's five crops are embedded with the adapted backbone, a logistic regression
is fitted, and held-out AUC is reported. The checkpoint's backbone is
"<name>+lora", so a loader that doesn't apply weight_deltas fails loudly
instead of scoring with the wrong weights.

Needs a rented GPU (docs/plan.md). The number of epochs is fixed in the config:
never pick an epoch by looking at held-out scores.

Usage:
    python model/steps/step5_lora_finetune.py --config model/configs/step5.yaml
"""
import math
import sys
import time
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from shared import model_interface as mi  # noqa: E402

LAYER_PATTERN = {"dinov2": "layer", "clip": "layers", "siglip": "layers"}


def lora_targets(model, kind):
    """Query/value projections, under whichever names this transformers version uses
    (DINOv2: "query"/"value" in 4.x, "q_proj"/"v_proj" in 5.x)."""
    leaves = {n.rsplit(".", 1)[-1] for n, _ in model.named_modules()}
    for names in (["q_proj", "v_proj"], ["query", "value"]):
        if set(names) <= leaves:
            return names, LAYER_PATTERN[kind]
    raise SystemExit(f"can't find query/value projections in the {kind} backbone")


class RandomCropImages(torch.utils.data.Dataset):
    """One random native-resolution 224 crop per image per epoch (same scale the recipe sees)."""

    def __init__(self, rows, root, mean, std, augment, seed):
        self.rows, self.root, self.mean, self.std = rows, Path(root), mean, std
        self.augment, self.seed, self.epoch = augment, seed, 0

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        name = self.rows[i]["filename"]
        rng = np.random.default_rng([self.seed, 1000 + self.epoch, zlib.crc32(name.encode())])
        with Image.open(self.root / name) as img:
            img = img.convert("RGB")
        if self.augment:
            img = common.random_degrade(img, rng)
        w, h = img.size
        if min(w, h) < mi.CROP:
            s = mi.CROP / min(w, h)
            img = img.resize((max(mi.CROP, round(w * s)), max(mi.CROP, round(h * s))), Image.BICUBIC)
            w, h = img.size
        x, y = int(rng.integers(0, w - mi.CROP + 1)), int(rng.integers(0, h - mi.CROP + 1))
        crop = img.crop((x, y, x + mi.CROP, y + mi.CROP))
        return mi.crops_to_tensor([crop], self.mean, self.std)[0], float(self.rows[i]["is_ai"])


def target_param_names(model, kind, layers):
    targets, pattern = lora_targets(model, kind)
    return [n for n, _ in model.named_parameters()
            if any(f".{pattern}.{i}." in f".{n}" for i in layers) and any(n.endswith(f"{t}.weight") for t in targets)]


def train_lora(cfg, train):
    from peft import LoraConfig, get_peft_model

    lc = cfg["lora"]
    seed = cfg["seed"]
    torch.manual_seed(seed)
    bb = mi.load_backbone(cfg["backbone"]["name"])
    model, kind, device = bb.model, bb.kind, bb.device
    n_layers = model.config.num_hidden_layers
    layers = list(range(n_layers - int(lc["last_n_blocks"]), n_layers))
    names = target_param_names(model, kind, layers)
    original = {n: p.detach().clone() for n, p in model.named_parameters() if n in set(names)}
    targets, pattern = lora_targets(model, kind)
    peft_model = get_peft_model(model, LoraConfig(r=int(lc["r"]), lora_alpha=int(lc["alpha"]),
                                                  lora_dropout=float(lc["dropout"]), target_modules=targets,
                                                  layers_to_transform=layers, layers_pattern=pattern, bias="none"))
    peft_model.print_trainable_parameters()
    bb.model = peft_model
    head = torch.nn.Linear(model.config.hidden_size, 1).to(device)

    ds = RandomCropImages(train, cfg["data"]["matched_dir"], bb.mean, bb.std, bool(lc.get("augment", True)), seed)
    batch = int(lc["batch_size"])
    steps_per_epoch = math.ceil(len(ds) / batch)
    total = int(lc.get("max_steps") or steps_per_epoch * int(lc["epochs"]))
    opt = torch.optim.AdamW([{"params": [p for p in peft_model.parameters() if p.requires_grad], "lr": float(lc["lr"])},
                             {"params": head.parameters(), "lr": float(lc["head_lr"])}],
                            weight_decay=float(lc.get("weight_decay", 0.01)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + math.cos(math.pi * min(s, total) / total)))
    use_amp = device.type == "cuda"
    amp_dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    step, t0 = 0, time.time()
    peft_model.train()
    while step < total:
        ds.epoch = step // steps_per_epoch
        g = torch.Generator().manual_seed(seed + ds.epoch)
        loader = torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=True, generator=g,
                                             num_workers=cfg["run"]["num_workers"], drop_last=len(ds) > batch)
        running = []
        for x, y in loader:
            x, y = x.to(device), y.float().to(device)
            with torch.autocast(device.type, dtype=amp_dtype, enabled=use_amp):
                loss = loss_fn(head(bb.features(x)).squeeze(1).float(), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            running.append(loss.item())
            step += 1
            if step % 50 == 0 or step == total:
                print(f"  step {step}/{total}  epoch {ds.epoch}  loss {np.mean(running[-50:]):.4f}  {time.time() - t0:.0f}s")
            if step >= total:
                break

    merged = peft_model.merge_and_unload()
    deltas = {n: (p.detach() - original[n]).to(torch.float16).cpu() for n, p in merged.named_parameters() if n in original}
    size = max(float(d.abs().max()) for d in deltas.values())
    if size == 0:
        raise SystemExit("LoRA training changed no weights; check the learning rate and target modules.")
    print(f"  {len(deltas)} adapted weight matrices in blocks {layers[0]}-{layers[-1]}, largest change {size:.2e}")
    summary = {"r": int(lc["r"]), "alpha": int(lc["alpha"]), "blocks": layers, "targets": targets,
               "steps": total, "batch_size": batch, "final_loss": float(np.mean(running[-50:])) if running else None}
    return deltas, summary


def main(config_path: str):
    cfg = common.load_config(config_path)
    train, held, held_out = common.load_rows(cfg)
    name = cfg["backbone"]["name"]
    if mi.BACKBONES[mi.base_name(name)]["kind"] not in LAYER_PATTERN:
        raise SystemExit(f"no LoRA targets defined for {name}")

    deltas, summary = train_lora(cfg, train)
    tuned = f"{mi.base_name(name)}+lora"
    copies = int(cfg["train"].get("augment_copies", 0))
    # Embeds with a FRESH backbone + saved deltas: the exact path load_checkpoint() takes.
    clf, res = common.fit_and_evaluate(cfg, train, held, common.embedding_features(cfg, tuned, deltas), copies, "step 5 LoRA")
    common.warn_results(res)
    res.update(step=5, backbone=tuned, lora=summary, augment_copies=copies,
               held_out_generators=sorted(held_out), n_held_out_images=len(held))
    return common.save(cfg, {"backbone": tuned, "classifier": clf, "weight_deltas": deltas, "lora": summary}, res)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    main(parser.parse_args().config)
