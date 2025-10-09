#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import random
from typing import List

import torch
from torch.utils.data import Dataset, DataLoader, Sampler


class SimpleDataset(Dataset):
    """
    一个最简单的数据集：返回 0..N-1 的整数样本，便于观察采样结果。
    """

    def __init__(self, length: int = 100):
        self.length = int(length)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return int(idx)


class DistributedKRepeatSampler(Sampler):
    """
    每个迭代：
      1) 全局（所有 replica 合并后的视角）随机抽取 m 个 unique 样本索引
      2) 将每个索引重复 k 次，得到 n*b 个条目（n=num_replicas，b=batch_size）
      3) 打散后，按 rank 切出当前进程的 b 个索引，作为一个 batch

    重点：
      - 该 Sampler 是“无限”的（__iter__ 中 while True）
      - 通过 set_epoch(epoch) 同步随机数种子，使多进程/多卡在相同 epoch 下抽到一致的全局顺序
    """

    def __init__(self, dataset, batch_size, k, num_replicas, rank, seed=0):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.k = int(k)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.seed = int(seed)

        # 每轮总条目数 = 世界大小 * 每卡 batch_size
        self.total_samples = self.num_replicas * self.batch_size
        assert self.k > 0, "k 必须 > 0"
        assert (
            self.total_samples % self.k == 0
        ), f"k({self.k}) 必须整除 n*b({self.total_samples})"

        # 本轮需要的 unique 样本数 m
        self.m = self.total_samples // self.k
        self.epoch = 0

    def __iter__(self):
        while True:
            # 同步随机性：不同 rank 但相同 epoch → 相同的全局随机序列
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)

            # 1) 选 m 个唯一索引
            indices = torch.randperm(len(self.dataset), generator=g)[: self.m].tolist()

            # 2) 每个索引重复 k 次（顺序先不打乱）
            repeated = [idx for idx in indices for _ in range(self.k)]

            # 3) 打散
            order = torch.randperm(len(repeated), generator=g).tolist()
            shuffled = [repeated[i] for i in order]

            # 4) 按 rank 切出当前卡的一个 batch
            start = self.rank * self.batch_size
            end = start + self.batch_size
            yield shuffled[start:end]

    def set_epoch(self, epoch):
        self.epoch = int(epoch)


def demo_once(num_replicas: int, batch_size: int, k: int, dataset_len: int, epoch: int):
    """单轮模拟：展示在同一个 epoch 下，不同 rank 各自拿到的 batch。"""
    ds = SimpleDataset(dataset_len)

    # 为每个 rank 建立各自的 sampler 和 dataloader
    samplers: List[DistributedKRepeatSampler] = []
    iters = []
    for rank in range(num_replicas):
        sp = DistributedKRepeatSampler(
            dataset=ds,
            batch_size=batch_size,
            k=k,
            num_replicas=num_replicas,
            rank=rank,
            seed=42,
        )
        sp.set_epoch(epoch)
        samplers.append(sp)
        iters.append(iter(DataLoader(ds, batch_sampler=sp, num_workers=0)))

    # 取每个 rank 的一个 batch 进行展示
    all_batches = []
    for rank in range(num_replicas):
        batch = next(iters[rank])  # tensor of indices
        all_batches.append((rank, batch))

    # 打印观测
    print(f"\nEpoch = {epoch}")
    print(
        f"设置：num_replicas={num_replicas}, batch_size={batch_size}, k={k}, dataset_len={dataset_len}"
    )
    total_samples = num_replicas * batch_size
    m = total_samples // k
    print(f"本轮应抽取的 unique 样本数 m = (n*b)/k = ({num_replicas}*{batch_size})/{k} = {m}")

    # 各 rank 的 batch
    for rank, batch in all_batches:
        print(f"  rank {rank} batch: {batch.tolist()}")

    # 合并后检查：
    merged: List[int] = sum([b.tolist() for _, b in all_batches], [])
    # 统计每个样本出现次数
    counts = {}
    for idx in merged:
        counts[idx] = counts.get(idx, 0) + 1

    unique_drawn = sorted(counts.keys())
    print(f"合并后全局 batch 大小 = {len(merged)} (应为 n*b = {total_samples})")
    print(f"合并后 unique 样本数量 = {len(unique_drawn)} (应为 m = {m})")
    # 随机展示几个样本的重复次数，应接近 k
    example_keys = random.sample(unique_drawn, k=min(5, len(unique_drawn))) if unique_drawn else []
    for key in example_keys:
        print(f"  样本 {key} 出现次数 = {counts[key]} (期望 ≈ k={k})")


def main():
    parser = argparse.ArgumentParser(description="DistributedKRepeatSampler 演示")
    parser.add_argument("--num_replicas", type=int, default=4, help="世界大小 n")
    parser.add_argument("--batch_size", type=int, default=3, help="每卡 batch 大小 b")
    parser.add_argument("--k", type=int, default=2, help="每个 unique 样本重复次数 k")
    parser.add_argument("--dataset_len", type=int, default=50, help="数据集长度 N")
    parser.add_argument("--epochs", type=int, default=2, help="演示的 epoch 次数")
    args = parser.parse_args()

    # 约束：n*b 必须能被 k 整除
    total = args.num_replicas * args.batch_size
    if total % args.k != 0:
        raise SystemExit(
            f"约束失败：n*b 必须能被 k 整除；当前 n*b={total}, k={args.k}"
        )

    for epoch in range(args.epochs):
        demo_once(
            num_replicas=args.num_replicas,
            batch_size=args.batch_size,
            k=args.k,
            dataset_len=args.dataset_len,
            epoch=epoch,
        )


if __name__ == "__main__":
    main()


