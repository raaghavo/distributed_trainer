import os, time, math, torch, torch.distributed as dist, torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from prometheus_client import Counter, Gauge, Histogram, start_http_server

# these are the metrics that I decided to track
LOSS = Gauge("train_loss", "Current training loss")
STEP_SEC = Histogram("step_seconds", "Step duration (s)")
SAMPLES = Counter("samples_total", "Total samples processed")

# setups up the process group for distributed training
def setup(rank, world_size, master_addr, master_port):
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

# destroys our process group (aka our distributed training group)
def cleanup():
    dist.destroy_process_group()

# saves a checkpoint of our model training
def save_ckpt(model, opt, step, path):
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step}, path)

# loads from a saved checkpoint
def load_ckpt(model, opt, path):
    if os.path.exists(path):
        ckpt = torch.load(path, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        return ckpt["step"]
    return 0

# sets up the model we are going to use
class ToyModel(torch.nn.Module):
    def __init__(self, d=1024): super().__init__(); self.l = torch.nn.Linear(d, 1)
    def forward(self, x): return self.l(x)

def run(rank, world_size):
    # Env from K8s
    master_addr = os.environ["MASTER_ADDR"]
    master_port = int(os.environ.get("MASTER_PORT", "29500"))
    work_dir = os.environ.get("CKPT_DIR", "/mnt/ckpt")
    total_steps = int(os.environ.get("TOTAL_STEPS", "500"))
    batch = int(os.environ.get("BATCH", "512"))

    # One metrics server per pod (port 9000 + rank to avoid clashes in single-container demo)
    if rank == 0:
        start_http_server(9000)

    setup(rank, world_size, master_addr, master_port)

    model = ToyModel().cpu()
    model = DDP(model)
    opt = torch.optim.SGD(model.parameters(), lr=1e-3)

    ckpt_path = os.path.join(work_dir, "ckpt.pt")
    step = load_ckpt(model, opt, ckpt_path)

    # synthetic data
    for s in range(step, total_steps):
        t0 = time.perf_counter()
        x = torch.randn(batch, 1024)
        y = torch.randn(batch, 1)
        pred = model(x); loss = torch.nn.functional.mse_loss(pred, y)
        opt.zero_grad(); loss.backward(); opt.step()
        dt = time.perf_counter() - t0

        # metrics (rank 0 to reduce noise)
        if dist.get_rank() == 0:
            LOSS.set(float(loss.item()))
            STEP_SEC.observe(dt)
            SAMPLES.inc(batch)

        # periodic checkpoint (rank 0 writes)
        if s % 50 == 0 and dist.get_rank() == 0:
            save_ckpt(model, opt, s, ckpt_path)

    if dist.get_rank() == 0:
        save_ckpt(model, opt, total_steps, ckpt_path)
    cleanup()

def main():
    world_size = int(os.environ["WORLD_SIZE"])
    mp.spawn(run, args=(world_size,), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()
