#!/usr/bin/env python3
"""Streaming re-implementation of brush-multi/scripts/extract_dino_features.py.

Same math, same on-disk contract (<data>/dino_features/{stem}.npy [h/14,w/14,pca_dim]
f32 + pca.npy [768,pca_dim] + meta.json), but bounded memory.

Why: the shipped script appends every image's [90,90,768] f32 map to a Python list
and torch.cat's them before fitting the PCA. That is 24.88 MB per view, so 5664
views = 141 GB resident. It cannot run on a 64 GB machine.

Deviations, all recorded in meta.json under "extraction_variant":
  * PCA basis = exact eigendecomposition of the covariance of centred features,
    accumulated streaming over a seeded random subset of --pca-fit-images images
    (all 8100 rows of each). The shipped script uses torch.pca_lowrank(niter=20)
    over <=4e6 rows sampled from ALL images. Same subspace, computed exactly
    rather than by randomized range-finding, from a comparable row count.
  * Projection is uncentred (feats @ basis), byte-identical in form to the
    shipped script's `feats.reshape(-1,d) @ pca_matrix`.
"""
import argparse, json, os, time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

NORMALIZE = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def get_img_resolution(H, W, max_size, p):
    if H < W:
        new_W = (max_size // p) * p
        new_H = (int((H / W) * max_size) // p) * p
    else:
        new_H = (max_size // p) * p
        new_W = (int((W / H) * max_size) // p) * p
    return new_H, new_W


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--pca-dim", type=int, default=96)
    ap.add_argument("--model", type=str, default="dinov2_vitb14")
    ap.add_argument("--max-size", type=int, default=1260)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--pca-fit-images", type=int, default=700)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir-name", type=str, default="dino_features")
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    img_dir = args.data / "images"
    paths = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not paths:
        raise SystemExit(f"no images in {img_dir}")

    dev = torch.device(
        args.device
        or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    )
    model = torch.hub.load("facebookresearch/dinov2", args.model).to(dev).eval()
    patch = model.patch_embed.patch_size[0]
    print(f"{len(paths)} images, device {dev}, patch {patch}", flush=True)

    def feats_for(path):
        """[h/14, w/14, 768] float32 on CPU -- identical math to the shipped script."""
        img = Image.open(path).convert("RGB")
        W, H = img.size
        h, w = get_img_resolution(H, W, args.max_size, patch)
        t = transforms.functional.to_tensor(img)
        t = transforms.functional.resize(
            t, (h, w), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True
        )
        t = NORMALIZE(t).to(dev)
        with torch.no_grad():
            d = model.get_intermediate_layers(t[None], reshape=True)[0]
            d = d.squeeze().permute(1, 2, 0) / 10
        return d.float().cpu(), (H, W)

    out_dir = args.data / args.out_dir_name
    out_dir.mkdir(exist_ok=True)
    pca_path = out_dir / "pca.npy"

    # ---- Pass A: streaming covariance over a seeded image subset -> exact PCA basis.
    if pca_path.is_file() and args.skip_existing:
        basis = torch.from_numpy(np.load(pca_path))
        feat_dim = basis.shape[0]
        n_fit_rows = -1
        print(f"reusing existing PCA basis {tuple(basis.shape)}", flush=True)
    else:
        rng = np.random.default_rng(args.seed)
        n_fit = min(args.pca_fit_images, len(paths))
        fit_idx = np.sort(rng.choice(len(paths), size=n_fit, replace=False))
        t0 = time.time()
        ssum = None
        gram = None
        n_fit_rows = 0
        for k, i in enumerate(fit_idx):
            f, _ = feats_for(paths[i])
            X = f.reshape(-1, f.shape[-1]).double()
            if gram is None:
                feat_dim = X.shape[1]
                gram = torch.zeros(feat_dim, feat_dim, dtype=torch.float64)
                ssum = torch.zeros(feat_dim, dtype=torch.float64)
            gram += X.T @ X
            ssum += X.sum(0)
            n_fit_rows += X.shape[0]
            if (k + 1) % 100 == 0:
                print(f"  pca-fit {k+1}/{n_fit}  {time.time()-t0:.0f}s", flush=True)
        mean = ssum / n_fit_rows
        cov = gram / n_fit_rows - torch.outer(mean, mean)
        cov = (cov + cov.T) / 2
        evals, evecs = torch.linalg.eigh(cov)
        order = torch.argsort(evals, descending=True)[: args.pca_dim]
        basis = evecs[:, order].float().contiguous()  # [768, pca_dim]
        var_kept = float(evals[order].sum() / evals.clamp(min=0).sum())
        print(
            f"PCA fit: {n_fit} images, {n_fit_rows} rows, basis {tuple(basis.shape)}, "
            f"variance retained {var_kept:.4f}, {time.time()-t0:.0f}s",
            flush=True,
        )
        np.save(pca_path, basis.numpy().astype(np.float32))

    # ---- Pass B: project every image and write.
    t0 = time.time()
    image_shape = None
    written = 0
    for i, path in enumerate(paths):
        dst = out_dir / f"{path.stem}.npy"
        if args.skip_existing and dst.is_file():
            continue
        f, (H, W) = feats_for(path)
        if image_shape is None:
            image_shape = [H, W]
        h, w, d = f.shape
        proj = (f.reshape(-1, d) @ basis).reshape(h, w, args.pca_dim)
        np.save(dst, np.ascontiguousarray(proj.numpy().astype(np.float32)))
        written += 1
        if written % 250 == 0:
            r = written / (time.time() - t0)
            print(
                f"  project {i+1}/{len(paths)}  {r:.2f} img/s  "
                f"eta {(len(paths)-i-1)/max(r,1e-9)/60:.1f} min",
                flush=True,
            )

    if image_shape is None:
        f, (H, W) = feats_for(paths[0])
        image_shape = [H, W]

    meta = {
        "model": args.model,
        "patch_size": patch,
        "pca_dim": args.pca_dim,
        "scale_div": 10,
        "max_size": args.max_size,
        "image_shape": image_shape,
        "extraction_variant": {
            "script": "extract_dino_features_streaming.py",
            "pca": "exact eigh of streaming covariance (centred), uncentred projection",
            "pca_fit_images": int(args.pca_fit_images),
            "pca_fit_rows": int(n_fit_rows),
            "seed": int(args.seed),
        },
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {written} feature maps to {out_dir} in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
