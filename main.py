import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
from UNetCond import UNetCond
from Diffuser import Diffuser
import numpy as np

def show_images(images, labels=None, rows=2, cols=10):
    fig = plt.figure(figsize=(cols, rows))
    i = 0
    for r in range(rows):
        for c in range(cols):
            ax = fig.add_subplot(rows, cols, i + 1)
            plt.imshow(images[i], cmap='gray')
            if labels is not None:
                ax.set_xlabel(labels[i].item())
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])
            i += 1
    plt.tight_layout()
    plt.show()

# Hyperparams
image_size = 28
batch_size = 128
num_timesteps = 1000
epochs = 10
lr = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Transform
transform = transforms.Compose([
    transforms.ToTensor()
])

# Dataset & Dataloader
dataset = torchvision.datasets.MNIST(
    root = './data',
    download = True,
    train = True,
    transform = transform
)
dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = True)

# Model & Optimizer
model = UNetCond(num_labels = 10)
model.to(device)
diffuser = Diffuser(num_timesteps = num_timesteps, device = device)
optimizer = torch.optim.Adam(model.parameters(), lr = lr)

# Train
losses = []
for epoch in range(epochs):
    # Iteration loss and counter
    total_loss = 0
    cnt = 0

    # Generating images for every epoch
    imgs, labels = diffuser.sample(model)
    show_images(images = imgs, labels = labels)

    for images, labels in tqdm(dataloader):

        optimizer.zero_grad()

        # x0s and labels
        x = images.to(device)
        labels = labels.to(device)

        t = torch.randint(1, num_timesteps + 1, (len(images),), device = device) # timestep sampling

        # No-condition training for 10%
        if np.random.random() < 0.1: labels = None

        # Add noise(forward process) & noise prediction(backward process)
        x_t, noise = diffuser.add_noise(x, t)
        noise_pred = model(x_t, t, labels)

        # loss (the mse loss between the noise added from x0 to xt and the noise that the model predicted, model input: xts, ts, labels)
        loss = F.mse_loss(noise, noise_pred)

        # Backprop
        loss.backward()

        # Step
        optimizer.step()

        total_loss += loss.item()
        cnt += 1
    
    avg_loss = total_loss / cnt
    losses.append(avg_loss)

    print(f"Epoch: {epoch} ||----------------------|| Loss: {avg_loss}")

plt.title("Epoch loss")
plt.plot(losses)
plt.xlabel("epoch")
plt.ylabel("loss")
plt.show()

# generate samples
images, labels = diffuser.sample(model)
show_images(images, labels)