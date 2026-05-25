import argparse, json, sys, os
import numpy as np
import torch
import torch.nn.functional as F

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(SCRIPT_DIR)
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

from guided_diffusion.script_util import (
    create_classifier, classifier_defaults, f_extractor_defaults,
)
from guided_diffusion import dist_util


def load_classifier(path, cfg):
    model = create_classifier(**cfg)
    state = dist_util.load_state_dict(path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz_path", required=True)
    parser.add_argument("--classifier_path", required=True)
    parser.add_argument("--f_extractor_path", required=True)
    parser.add_argument("--class_id", type=int, default=99)
    args = parser.parse_args()

    data = np.load(args.npz_path)
    images_uint8 = data["arr_0"]
    n = images_uint8.shape[0]

    images = torch.from_numpy(images_uint8).float()
    images = images.permute(0, 3, 1, 2)
    images = images / 127.5 - 1.0

    f_cfg = f_extractor_defaults()
    f_extractor = load_classifier(args.f_extractor_path, f_cfg)

    clf_cfg = {**classifier_defaults(), "image_size": 8, "in_channels": 512, "out_channels": 100}
    classifier = load_classifier(args.classifier_path, clf_cfg)

    device = torch.device("cpu")
    f_extractor.to(device)
    classifier.to(device)

    t_zeros = torch.zeros(n, dtype=torch.long, device=device)
    labels = torch.full((n,), args.class_id, dtype=torch.long, device=device)

    confidences, losses = [], []
    with torch.no_grad():
        for i in range(n):
            img = images[i:i+1].to(device)
            t = t_zeros[i:i+1]
            label = labels[i:i+1]
            latents = f_extractor(img, timesteps=t, f_out=True)
            logits = classifier(latents, timesteps=t)
            probs = F.softmax(logits, dim=-1)
            conf = probs[0, args.class_id].item()
            loss = F.cross_entropy(logits, label).item()
            confidences.append(conf)
            losses.append(loss)

    print(json.dumps({
        "n": n,
        "classifier_mean_confidence": float(np.mean(confidences)),
        "classifier_mean_loss": float(np.mean(losses)),
        "confidences": confidences,
        "losses": losses,
    }))


if __name__ == "__main__":
    main()
