import os
import json
import csv
from typing import List, Dict

import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio
from diffusers import StableAudioPipeline
from diffusers.optimization import get_scheduler

class AudioTextDataset(Dataset):
    """Simple dataset reading paths and prompts from a CSV file."""

    def __init__(self, csv_path: str, sample_rate: int):
        self.sample_rate = sample_rate
        self.items: List[Dict[str, str]] = []
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.items.append(row)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        item = self.items[idx]
        audio, sr = torchaudio.load(item["path"])
        if sr != self.sample_rate:
            audio = torchaudio.functional.resample(audio, sr, self.sample_rate)
        prompt = item["prompt"]
        return {"audio": audio, "prompt": prompt}

def encode_audio(pipe: StableAudioPipeline, audio: torch.Tensor) -> torch.Tensor:
    audio = audio.to(pipe._execution_device)
    with torch.no_grad():
        latents = pipe.vae.encode(audio).latent_dist.sample()
        latents = latents * pipe.vae.config.scaling_factor
    return latents

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    with open("model_config.json") as f:
        config = json.load(f)
    sample_rate = config["sample_rate"]

    # Load pretrained components from current directory
    pipe = StableAudioPipeline.from_pretrained(".")
    pipe = pipe.to(device)

    # Example dataset CSV expected to have columns: `path,prompt`
    train_dataset = AudioTextDataset("train_dataset.csv", sample_rate)
    dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)

    optimizer = torch.optim.AdamW(pipe.transformer.parameters(), lr=5e-5)
    num_training_steps = 1000
    lr_scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=100,
        num_training_steps=num_training_steps,
    )

    pipe.transformer.train()
    for step, batch in enumerate(dataloader):
        if step >= num_training_steps:
            break

        prompts = batch["prompt"]
        audio = batch["audio"]
        latents = encode_audio(pipe, audio)

        noise = torch.randn_like(latents)
        timesteps = torch.randint(0, pipe.scheduler.config.num_train_timesteps, (latents.size(0),), device=device).long()
        noisy_latents = pipe.scheduler.add_noise(latents, noise, timesteps)

        text_inputs = pipe.tokenizer(
            list(prompts), return_tensors="pt", padding="max_length", truncation=True, max_length=128
        ).to(device)
        encoder_hidden_states = pipe.text_encoder(**text_inputs).last_hidden_state

        model_pred = pipe.transformer(noisy_latents, timesteps, encoder_hidden_states).sample
        loss = torch.nn.functional.mse_loss(model_pred, noise)

        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        if step % 50 == 0:
            print(f"step {step}: loss {loss.item():.4f}")

    pipe.save_pretrained("finetuned_model")

if __name__ == "__main__":
    main()
