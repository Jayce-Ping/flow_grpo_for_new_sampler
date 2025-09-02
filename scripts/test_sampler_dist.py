# This is a test script for the distributed K-repeat sampler, simplified from train_flux.py with only dataloading and gathering parts.
import time
import argparse
import hashlib
import math
from collections import Counter
from itertools import groupby
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

    # It will cause misalignment if we use accelerator to prepare dataloader???
    # Maybe our DistSampler and accelerator's preparation make something duplicate for distribution preparations.
    # train_dataloader = accelerator.prepare(train_dataloader) 

    if accelerator.is_main_process:
        print(f"[world_size={accelerator.num_processes}] "
              f"dataset={len(train_dataset)} k={train_sampler.k} m_final={train_sampler.m} "
              f"num_batches_per_epoch={num_batches_per_epoch} bs={args.batch_size}")


    it = iter(train_dataloader) # The position of this assignment seems not the real cause
    epoch_criterion = {}
    all_group_size = set()
    # 4) Distributed testing: count unique prompt per epoch from all processes
    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)

        local_counter = Counter()
        global_counter = Counter() if accelerator.is_main_process else None

        # if accelerator.is_main_process:
            # Sleep for 3 seconds to simulate async data loading
            # time.sleep(3)

        all_tensors = []
        for _ in range(num_batches_per_epoch):
            prompts, prompt_metadata = next(it)  # prompts: List[str]

            # Map prompts to fixed-length 64-bit hashes for gathering
            local_hashes = [prompt_hash64(p) for p in prompts]
            local_counter.update(local_hashes)

            h = torch.tensor(local_hashes, dtype=torch.long, device=accelerator.device)
            all_tensors.append(h)
        
        all_tensors = torch.cat(all_tensors, dim=0)
        gathered = accelerator.gather(all_tensors).cpu().tolist()

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
            criterion = {
                "global_unique error": global_unique != train_sampler.m,
                "global_total error": global_total != train_sampler.k * train_sampler.m,
                "group_size error": not all(v == args.k for v in global_counter.values())
            }
            epoch_criterion[epoch] = criterion
            all_group_size.update(set(global_counter.values()))

            if False:
                # Print detailed info for each epoch
                print(
                    f"Epoch {epoch} | "
                    f"global_unique={global_unique} (expected {train_sampler.m}) | "
                    f"global_total={global_total} (expected {train_sampler.k * train_sampler.m}) | "
                    f"local_unique(proc0)={local_unique} local_total(proc0)={local_total} | "
                    f"All group has size of {args.k} :({all(v==args.k for v in global_counter.values())}) | "
                    f"All group sizes (expected all={args.k}): {list(global_counter.values())}"
                    "\n"
                )
        else:
            pass
            # Print local statistics for other processes (optional) - maybe will mess up the screen
            # print(f"Rank {accelerator.process_index} Epoch {epoch} | local_unique={local_unique} local_total={local_total}\n")

    if accelerator.is_main_process:
        print("Done.")
        group_by_error = groupby(epoch_criterion.items(), lambda x: tuple(x[1].values()))
        for error_boolean, group in group_by_error:
            epoches = [epoch for epoch, _ in group]
            if any(error_boolean):
                e = ", ".join(k for k, v in zip(epoch_criterion[0].keys(), error_boolean) if v)
                print(f"{e} with {len(epoches)} epoches: {epoches}")
            else:
                print(f"No error for {len(epoches)} epoches: {epoches}.")

        print(f"[all_group_size = {all_group_size}.]")



if __name__ == "__main__":
    main()