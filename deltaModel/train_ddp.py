"""
Distributed Data Parallel (DDP) training example with PyTorch.

circa 25 Jan 2025
"""
import os
import time
import logging
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset
from torch.nn.parallel import DistributedDataParallel as DDP

from core.utils import set_randomseed
set_randomseed()

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

#### Configuration & synthetic data
###################################
# Dataset parameters
input_size = 1000
output_size = 2
num_samples = 10000
batch_size = 10000

# Hyperparameters
gpu_ids = [6, 7]  # GPU IDs to use
world_size = 2  # Number of GPUs
epochs = 2
hidden_size = 256
lr = 0.001

# Synthetic dataset
# Random input features and labels
x = torch.rand(num_samples, input_size)
y = torch.randint(0, output_size, (num_samples,))

# Create DataLoader
dataset = TensorDataset(x, y)
###################################

class ToyModel(nn.Module):
    """Simple linear model for demonstration."""
    def __init__(self, input_size, hidden_size, output_size):
        super(ToyModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
    

def setup(rank, world_size):
    """Setup distributed training environment."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    # Map ranks to specific GPU IDs
    gpus = gpu_ids  # GPUs to use (rank 0 -> GPU 6, rank 1 -> GPU 7)
    torch.cuda.set_device(gpus[rank])  # Set the device for each rank

    log.info(f"Process {rank} initialized.")


def cleanup():
    """Cleanup the distributed environment."""
    try:
        dist.destroy_process_group()
        # log.info("Distributed process group destroyed")
    except Exception as e:
        log.error(f"Cleanup failed: {e}")


def train_ddp(
    rank,
    world_size,
    dataset,
    epochs,
    input_size,
    output_size,
    hidden_size,
    batch_size,
    lr,
):
    """Distributed Data Parallel training example."""
    setup(rank, world_size)

    model = ToyModel(input_size, hidden_size, output_size).to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    # Sampler, loss, optimizer
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(ddp_model.parameters(), lr=lr)

    # Initialize timers
    start_time = time.time()
    t_total = 0.0
    t_epoch = 0.0
    t_back_total = 0.0

    for epoch in range(epochs):
        sampler.set_epoch(epoch)
        ddp_model.train()

        epoch_loss = 0.0
        t_ep0 = time.time()
        
        for batch in dataloader:
            x_batch, y_batch = batch
            x_batch, y_batch = x_batch.to(rank), y_batch.to(rank)

            optimizer.zero_grad()
            predictions = ddp_model(x_batch)
            loss = criterion(predictions, y_batch)

            t_back0 = time.time()
            loss.backward()            
            optimizer.step()
            t_back = time.time() - t_back0

            epoch_loss += loss.item()
            t_back_total += t_back

        t_epoch += time.time() - t_ep0

        # After each epoch, use all_reduce to gather total time across all processes
        dist.all_reduce(torch.tensor(t_epoch).to(rank), op=dist.ReduceOp.SUM)
        dist.all_reduce(torch.tensor(t_back_total).to(rank), op=dist.ReduceOp.SUM)

        # Log times per epoch
        log.info(f"Rank {rank}, Epoch {epoch + 1}, Loss: {epoch_loss / len(dataloader)}")

    # After all epochs, aggregate the total time and calculate the averages
    dist.barrier()  # Ensure all ranks finish before logging results
    if rank == 0:  # Log only from rank 0 to avoid duplicated messages
        # Average times per epoch and per backward pass across all processes
        avg_t_epoch = t_epoch / epochs
        avg_t_back = t_back_total / (epochs * len(dataloader))

        # Total runtime
        t_total = time.time() - t_ep0  # Total time taken by training
        print(f"------ \n Training: {t_total:.4f} seconds.")
        print(f"------ \n Average time per epoch: {avg_t_epoch:.4f} seconds.")
        print(f"------ \n Average time per backward: {avg_t_back:.4f} seconds.")

    cleanup()



if __name__ == "__main__":
    if world_size < 2:
        log.error("DDP requires at least 2 GPUs.")
    else:
        print("\n=== DistributedDataParallel Training ===")
        
        import torch.multiprocessing as mp

        t0 = time.time()
        mp.spawn(
            train_ddp,
            args=(world_size, dataset, epochs, input_size, output_size, hidden_size, batch_size, lr),
            nprocs=world_size,
            join=True,
        )
        t = time.time() - t0
        print(f"-------- \n Total Training Time: {t:.6f}s")
