from dataclasses import dataclass
import typing as t

kuzushiji_dir = "/store/train.csv"
plot_dir = "/store/plot"
image_dir = "/store/images"
lr = 1e-4

n_splits = 4
seed = 777
device = "cuda"

max_size = 512
batch_size = 8
num_workers = 8
