from dataclasses import dataclass
import typing as t

kuzushiji_dir = "/store/train.csv"
plot_dir = "/store/plot"
image_dir = "/store/images"
lr = 1e-3

n_splits = 4
seed = 777
device = "cuda"
hidden_channels = 32
out_idx = 5

max_size = 1024
batch_size = 2
num_workers = 8
