"""
windowed_classifier_sample.py
Extends classifier_sample.py with --t_start / --t_end flags that gate
minority guidance to a contiguous window of the denoising chain.
Guidance applied only when: t_start <= current_timestep < t_end
"""

import argparse
import os
import sys

import numpy as np
import torch as th
import torch.distributed as dist
import torch.nn.functional as F

from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    classifier_defaults,
    create_model_and_diffusion,
    create_classifier,
    add_dict_to_argparser,
    args_to_dict,
    f_extractor_defaults,
)


def _dist_ready():
    return dist.is_available() and dist.is_initialized()

def _world_size():
    return dist.get_world_size() if _dist_ready() else 1

def _rank():
    return dist.get_rank() if _dist_ready() else 0


def main():
    args = create_argparser().parse_args()
    th.set_num_threads(8)

    if args.seed != "":
        th.manual_seed(int(args.seed))
        np.random.seed(int(args.seed))

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    args.image_size = args.latent_size
    logger.log("loading classifier...")
    classifier = create_classifier(**args_to_dict(args, classifier_defaults().keys()))
    classifier.load_state_dict(
        dist_util.load_state_dict(args.classifier_path, map_location="cpu")
    )
    classifier.to(dist_util.dev())
    if args.classifier_use_fp16:
        classifier.convert_to_fp16()
    classifier.eval()

    args.image_size = f_extractor_defaults()["image_size"]
    logger.log("loading feature extractor...")
    f_extractor = create_classifier(**f_extractor_defaults())
    f_extractor.load_state_dict(
        dist_util.load_state_dict(args.f_extractor_path, map_location="cpu")
    )
    f_extractor.to(dist_util.dev())
    if args.classifier_use_fp16:
        f_extractor.convert_to_fp16()
    f_extractor.eval()

    t_start = args.t_start
    t_end = args.t_end
    logger.log(f"guidance window: t in [{t_start}, {t_end}), scale={args.classifier_scale}")

    def cond_fn(x, t, y=None):
        assert y is not None
        current_t = int(t[0].item())
        if not (t_start <= current_t < t_end):
            return th.zeros_like(x)
        with th.enable_grad():
            x_in = x.detach().requires_grad_(True)
            latents = f_extractor(x_in, timesteps=t, f_out=True)
            logits = classifier(latents, timesteps=t)
            log_probs = F.log_softmax(logits, dim=-1)
            selected = log_probs[range(len(logits)), y.view(-1)]
            return th.autograd.grad(selected.sum(), x_in)[0] * args.classifier_scale

    def model_fn(x, t, y=None):
        assert y is not None
        return model(x, t, y if args.class_cond else None)

    logger.log("sampling...")
    all_images = []
    all_labels = []
    while len(all_images) * args.batch_size < args.num_samples:
        model_kwargs = {}
        if args.use_manual_class:
            classes = args.manual_class_id * th.ones(
                size=(args.batch_size,), device=dist_util.dev()
            ).long()
        else:
            classes = th.randint(
                low=args.rand_starting_class_id,
                high=args.num_classes,
                size=(args.batch_size,),
                device=dist_util.dev(),
            )
        model_kwargs["y"] = classes
        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        sample = sample_fn(
            model_fn,
            (args.batch_size, 3, args.image_size, args.image_size),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            cond_fn=cond_fn,
            device=dist_util.dev(),
        )
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()

        if _dist_ready():
            gathered_samples = [th.zeros_like(sample) for _ in range(_world_size())]
            dist.all_gather(gathered_samples, sample)
            all_images.extend([s.cpu().numpy() for s in gathered_samples])
            gathered_labels = [th.zeros_like(classes) for _ in range(_world_size())]
            dist.all_gather(gathered_labels, classes)
            all_labels.extend([l.cpu().numpy() for l in gathered_labels])
        else:
            all_images.append(sample.cpu().numpy())
            all_labels.append(classes.cpu().numpy())
        logger.log(f"created {len(all_images) * args.batch_size} samples")

    arr = np.concatenate(all_images, axis=0)[: args.num_samples]
    label_arr = np.concatenate(all_labels, axis=0)[: args.num_samples]
    if _rank() == 0:
        shape_str = "x".join([str(x) for x in arr.shape])
        out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}.npz")
        logger.log(f"saving to {out_path}")
        np.savez(out_path, arr, label_arr)

    if _dist_ready():
        dist.barrier()
    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=10000,
        batch_size=16,
        use_ddim=False,
        model_path="",
        classifier_path="",
        classifier_scale=1.0,
        seed="",
        num_classes=100,
        use_manual_class=False,
        manual_class_id=0,
        rand_starting_class_id=0,
        f_extractor_path="",
        latent_size=8,
        t_start=0,
        t_end=1000,
    )
    defaults.update(model_and_diffusion_defaults())
    defaults.update(classifier_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
