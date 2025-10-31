import argparse
import os
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import tqdm
import sys
import json
import numpy as np
import io

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


class ConditionalEncoder(nn.Module):
    def __init__(self, input_channels, num_classes, latent_dim):
        super(ConditionalEncoder, self).__init__()
        self.conv1 = nn.Conv2d(input_channels + num_classes, 32, kernel_size=4, stride=2, padding=1)  # 28->14
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)  # 14->7
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0)  # 7->5
        self.flatten = nn.Flatten()
        self.fc_mu = nn.Linear(128*5*5, latent_dim)
        self.fc_logvar = nn.Linear(128*5*5, latent_dim)

    def forward(self, x, labels):
        labels = labels.view(labels.size(0), labels.size(1), 1, 1)
        labels = labels.expand(labels.size(0), labels.size(1), x.size(2), x.size(3))
        x = torch.cat((x, labels), dim=1)
        h = torch.relu(self.conv1(x))
        h = torch.relu(self.conv2(h))
        h = torch.relu(self.conv3(h))
        h = self.flatten(h)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


class ConditionalDecoder(nn.Module):
    def __init__(self, latent_dim, num_classes, output_channels):
        super(ConditionalDecoder, self).__init__()
        self.fc = nn.Linear(latent_dim + num_classes, 128*5*5)
        self.deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=0)            # 5->7
        self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)             # 7->14
        self.deconv3 = nn.ConvTranspose2d(32, output_channels, kernel_size=4, stride=2, padding=1) # 14->28

    def forward(self, z, labels):
        z = torch.cat((z, labels), dim=1)
        h = self.fc(z)
        h = h.view(-1, 128, 5, 5)
        h = torch.relu(self.deconv1(h))
        h = torch.relu(self.deconv2(h))
        x_reconstructed = torch.sigmoid(self.deconv3(h))
        return x_reconstructed


class CVAE(nn.Module):
    def __init__(self, input_channels=1, num_classes=10, latent_dim=120):
        super(CVAE, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = ConditionalEncoder(input_channels, num_classes, latent_dim)
        self.decoder = ConditionalDecoder(latent_dim, num_classes, input_channels)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, labels):
        mu, logvar = self.encoder(x, labels)
        z = self.reparameterize(mu, logvar)
        x_reconstructed = self.decoder(z, labels)
        return x_reconstructed, mu, logvar


def loss_function(x_reconstructed, x, mu, logvar):
    reconstruction_loss = nn.functional.binary_cross_entropy(x_reconstructed, x, reduction='sum')
    kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return reconstruction_loss + kl_divergence


def one_hot(labels, num_classes, device):
    if labels.dtype == torch.int64 or labels.dtype == torch.long:
        return torch.eye(num_classes, device=device)[labels]
    else:
        return labels.to(device).float()


def train(args):
    use_cuda = args.num_gpus > 0
    kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}
    device = torch.device('cuda' if use_cuda else 'cpu')

    logger.info("Iniciando treinamento....")
    train_loader = _get_train_data_loader(args.batch_size, args.data_dir, **kwargs)
    logger.info("Dataset carregado")

    model = CVAE(input_channels=1, num_classes=args.num_classes, latent_dim=args.latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    logger.info("Modelo Carregado")
    model.train()

    for epoch in tqdm.tqdm(range(args.epochs)):
        train_loss = 0
        for batch_idx, (data, labels) in enumerate(train_loader):
            data = data.to(device)
            labels = one_hot(labels, args.num_classes, device)

            optimizer.zero_grad()
            x_reconstructed, mu, logvar = model(data, labels)
            assert x_reconstructed.shape == data.shape, f"{x_reconstructed.shape} vs {data.shape}"

            loss = loss_function(x_reconstructed, data, mu, logvar)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        logger.info(f"Epoch [{epoch+1}/{args.epochs}], Loss: {train_loss/len(train_loader.dataset):.4f}")

    save_model(model, args.model_dir)


def save_model(model, model_dir):
    logger.info("Saving the model.")
    os.makedirs(model_dir, exist_ok=True)
    torch.save(model.cpu().state_dict(), os.path.join(model_dir, "model.pth"))
    torch.save(model.decoder.state_dict(), os.path.join(model_dir, "decoder.pth"))


def _get_train_data_loader(batch_size, training_dir, **kwargs):
    dataset = datasets.MNIST(training_dir, train=True, download=True, transform=transforms.ToTensor())
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, **kwargs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--latent_dim", type=int, default=120)
    parser.add_argument("--batch-size", dest='batch_size', type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--model-dir", type=str, default="./model")
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--num-gpus", type=int, default=0)
    args = parser.parse_args()
    train(args)
