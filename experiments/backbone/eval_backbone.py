import torch
from mipcandy import SimpleDataset, convert_logits_to_ids, dice_similarity_coefficient, convert_ids_to_logits, \
    NNUNetDataset
from torch import nn

from experiments.vars import ExpConfig


def eval_backbone(cfg: ExpConfig) -> None:
    output_dataset = SimpleDataset(f"{cfg.project_weights}/out{cfg.fold}", False)
    gt_dataset = cfg.dataset("Ts")
    with torch.no_grad():
        ces = []
        dices = []
        for idx in range(len(output_dataset)):
            # convert to class ids
            outputs = output_dataset[idx].unsqueeze(0)
            outputs = convert_logits_to_ids(outputs.softmax(1))
            labels = gt_dataset.label(idx).unsqueeze(0)
            # convert to one-hots
            outputs = convert_ids_to_logits(outputs, cfg.num_classes)
            labels = convert_ids_to_logits(labels, cfg.num_classes)
            # eval metrics
            ce = nn.functional.cross_entropy(outputs, labels)
            ces.append(ce)
            dice = dice_similarity_coefficient(outputs, labels)
            dices.append(dice)
            print(f"Case {idx}" + "=" * 10)
            print(f"CE: {ce:.4f}")
            print(f"Dice: {dice:.4f}")
        print(f"Average Dice: {sum(dices) / len(output_dataset):.4f}")
