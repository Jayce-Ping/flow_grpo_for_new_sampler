# This is a test script for the distributed K-repeat sampler, simplified from train_flux.py with only dataloading and gathering parts.

import argparse
import hashlib
import math
from collections import Counter
from accelerate.utils import set_seed

import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader

from flow_grpo.sampler import TextPromptDataset, DistributedKRepeatSampler

def prompt_hash64(s: str) -> int:
    d = hashlib.sha256(s.encode("utf-8")).digest()
    val = int.from_bytes(d[:8], byteorder="big", signed=False)
    # clamp to signed int64 range (63-bit) to avoid overflow in torch.long
    return val & ((1 << 63) - 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Dataset root")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--k", type=int, required=True, help="num_image_per_prompt")
    parser.add_argument("--m", type=int, required=True, help="unique_sample_num_per_epoch (least)")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=8)
    args = parser.parse_args()

    accelerator = Accelerator()
    set_seed(args.seed, device_specific=True)

    # 1) Create dataloader
    train_dataset = TextPromptDataset(args.dataset, "train")
    collate_fn = TextPromptDataset.collate_fn

    # 2) Create sampler for distributed training
    train_sampler = DistributedKRepeatSampler(
        dataset=train_dataset,
        batch_size=args.batch_size,
        k=args.k,
        m=args.m,
        num_replicas=accelerator.num_processes,
        rank=accelerator.process_index,
        seed=args.seed,
    )

    # basic consistency check
    assert train_sampler.m <= len(train_dataset), \
        f"sampler.m({train_sampler.m}) > len(dataset)({len(train_dataset)}), Please provide smaller m or smaller k to fit, or increase the dataset size."
    # Update num_batches_per_epoch
    num_batches_per_epoch = train_sampler.num_batches_per_epoch

    # 3) Create DataLoader (no shuffle, batches controlled by sampler)
    train_dataloader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=args.num_workers,
        collate_fn=collate_fn
    )
    accelerator.state.deepspeed_plugin.deepspeed_config['train_micro_batch_size_per_gpu'] = args.batch_size

    train_dataloader = accelerator.prepare(train_dataloader)

    if accelerator.is_main_process:
        print(f"[world_size={accelerator.num_processes}] "
              f"dataset={len(train_dataset)} k={train_sampler.k} m_final={train_sampler.m} "
              f"num_batches_per_epoch={num_batches_per_epoch} bs={args.batch_size}")

    error_epoch = {}
    # 4) Distributed testing: count unique prompt per epoch from all processes
    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)
        it = iter(train_dataloader)

        local_counter = Counter()
        global_counter = Counter() if accelerator.is_main_process else None

        for _ in range(num_batches_per_epoch):
            prompts, prompt_metadata = next(it)  # prompts: List[str]

            # Map prompts to fixed-length 64-bit hashes for gathering
            local_hashes = [prompt_hash64(p) for p in prompts]
            local_counter.update(local_hashes)

            h = torch.tensor(local_hashes, dtype=torch.long, device=accelerator.device)
            gathered = accelerator.gather(h).cpu().tolist()

            if accelerator.is_main_process:
                global_counter.update(gathered)

        # 5) Print statistics
        local_unique = len(local_counter.keys())
        local_total = sum(local_counter.values())
        if accelerator.is_main_process:
            global_unique = len(global_counter.keys())
            global_total = sum(global_counter.values())
            # Expected: global_unique == train_sampler.m; global_total == k*m
            # Record if unexpected case:
            if global_unique != train_sampler.m or global_total != train_sampler.k * train_sampler.m:
                error_epoch[epoch] = (global_unique, global_total)
            print(
                f"Epoch {epoch} | "
                f"global_unique={global_unique} (expected {train_sampler.m}) | "
                f"global_total={global_total} (expected {train_sampler.k * train_sampler.m}) | "
                f"local_unique(proc0)={local_unique} local_total(proc0)={local_total}"
                "\n"
            )
        else:
            pass
            # Print local statistics for other processes (optional) - maybe will mess up the screen
            # print(f"Rank {accelerator.process_index} Epoch {epoch} | local_unique={local_unique} local_total={local_total}\n")

    if accelerator.is_main_process:
        print("Done.")
        if len(error_epoch) > 0:
            for k, v in error_epoch.items():
                print(f"  Error at epoch {k}: global_unique={v[0]} global_total={v[1]}")
        else:
            print("No errors found.")



if __name__ == "__main__":
    main()