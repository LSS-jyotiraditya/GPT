# imports
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW, lr_scheduler
import os

from datasets import load_from_disk
from tokenizers import Tokenizer
from transformers import BitsAndBytesConfig

from tqdm.auto import tqdm
from moe_nano_gpt_model import NanoGPTMoE
# -----------------------------

# setup cuda
torch.cuda.memory._record_memory_history()
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_device("cpu")  # Keep default as CPU
print(f"using {device}")
# -----------------------------

# tokenizer (for vocab size)
tokenizer_file = "/root/stories/slms/data/TinyStories-tokenizer.json"
tokenizer = Tokenizer.from_file(tokenizer_file)
# -----------------------------

# hyperparameters
hyperparameters = {
    "n_epochs": 20,
    "vocab_size": tokenizer.get_vocab_size(),
    "batch_size": 32,
    "block_size": 1080,
    "learning_rate": 4e-2, # experiment with this more than anything else
    "n_embed": 256,
    "n_heads": 8,
    "n_layers": 12,
    "dropout": 0.1,
    "n_experts": 8,
    "top_k": 2,
    "use_4bit": True,  # Enable 4-bit quantization
}

n_epochs = hyperparameters['n_epochs']
batch_size = hyperparameters['batch_size']
block_size = hyperparameters['block_size']
learning_rate = hyperparameters['learning_rate']
use_4bit = hyperparameters['use_4bit']
# -----------------------------

# load pre-tokenized dataset
tokenized_dataset_path = "/root/stories/slms/data"
tokenized_data = load_from_disk(tokenized_dataset_path).with_format("torch")

train_ids = tokenized_data['train']
val_ids = tokenized_data['validation']
# -----------------------------

# setup model and trainer
if use_4bit and device == "cuda":
    # 4-bit quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    
    # Create model with quantization
    model = NanoGPTMoE(hyperparameters, device).to(device)
    print("Using 4-bit quantization")
else:
    model = NanoGPTMoE(hyperparameters, device).to(device)
    print("Using full precision")

train_dataloader = DataLoader(train_ids, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_ids, batch_size=batch_size, shuffle=False)

optimizer = AdamW(model.parameters(), lr=learning_rate)
num_params = sum(p.numel() for p in model.parameters())/1e6

num_training_steps = n_epochs * len(train_dataloader)
scheduler = lr_scheduler.OneCycleLR(optimizer=optimizer, max_lr=learning_rate, total_steps=num_training_steps)
print(f"{num_params:.3f}M parameters")
os.makedirs("checkpoints", exist_ok=True)
# -----------------------------

# load checkpoint (optional resume)
checkpoint_file = "checkpoints/moe-0.000M-checkpoint-0.pt"
# checkpoint = torch.load(checkpoint_file, map_location=device)
# model.load_state_dict(checkpoint['model'])
# optimizer.load_state_dict(checkpoint['optimizer'])
# scheduler.load_state_dict(checkpoint['scheduler'])
# saved_epoch = checkpoint['epoch']

saved_epoch = None
# -----------------------------

# train model
lossi = []
lri = []
progress_bar = tqdm(range(num_training_steps))

for epoch in range(n_epochs):
    model.train() # switch model to training mode
    if saved_epoch != None and epoch <= saved_epoch:
        continue

    for batch in train_dataloader:
        batch = batch['input_ids'].to(device)
        targets = torch.concat((batch[:, 1:], 2 * torch.ones([batch.shape[0], 1]).to(device)), dim=1).long()
        logits, loss = model(batch, targets)

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
        scheduler.step()
        progress_bar.update(1)

        if (progress_bar.n % 500 == 0):
            print(f"erratic train loss: {loss.item()} lr: {optimizer.param_groups[0]['lr']}")
        lossi.append(loss.log10().item())
        lri.append(optimizer.param_groups[0]['lr'])

    with torch.no_grad():
        # evaluate validation loss
        model.eval() # switch model to evaluation mode
        losses =  torch.zeros(len(val_dataloader), device=device)
        k = 0
        for batch in val_dataloader:
            batch = batch['input_ids'].to(device)
            targets = torch.concat((batch[:, 1:], 2 * torch.ones([batch.shape[0], 1]).to(device)), dim=1).long()
            logits, loss = model(batch, targets)

            losses[k] = loss.item()
            predictions = torch.argmax(logits, dim=-1)
            k += 1

        avg_val_loss = losses.mean()
        print(f"val loss: {avg_val_loss}")
        # -----------------------------

        # evaluate training loss
        losses =  torch.zeros(len(val_dataloader), device=device)
        k = 0
        for batch in train_dataloader:
            batch = batch['input_ids'].to(device)
            targets = torch.concat((batch[:, 1:], 2 * torch.ones([batch.shape[0], 1]).to(device)), dim=1).long()
            logits, loss = model(batch, targets)

            losses[k] = loss.item()
            predictions = torch.argmax(logits, dim=-1)
            k += 1

            if(k == len(val_dataloader)):
                break

        avg_train_loss = losses.mean()
        print(f"train loss: {avg_train_loss}")
        # -----------------------------

        # save checkpoint
        checkpoint = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "hyperparameters": hyperparameters,
            "val_loss": avg_val_loss,
            "train_loss": avg_train_loss.item(),
        }
        torch.save(checkpoint, f"checkpoints/moe-{num_params:.3f}M-checkpoint-{epoch}.pt")
        # -----------------------------
# -----------------------------

from matplotlib import pyplot as plt
plt.plot(torch.tensor(lossi).view(-1, 25).mean(1))
