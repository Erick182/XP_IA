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

        # concat image channels + one-hot label channels
        self.conv1 = nn.Conv2d(input_channels + num_classes, 32, kernel_size=4, stride=2, padding=1)  # [batch_size, 32, 14, 14]
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)  # [batch_size, 64, 7, 7]
        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)  # [batch_size, 128, 3, 3]

        self.flatten = nn.Flatten()  # [batch_size, 128*3*3]

        self.fc_mu = nn.Linear(128 * 3 * 3, latent_dim)
        self.fc_logvar = nn.Linear(128 * 3 * 3, latent_dim)

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

        self.fc = nn.Linear(latent_dim + num_classes, 128 * 3 * 3)
        self.deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)  # [batch_size, 64, 6, 6]
        self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=0)  # [batch_size, 32, 14, 14]
        self.deconv3 = nn.ConvTranspose2d(32, output_channels, kernel_size=4, stride=2, padding=1)  # [batch_size, 1, 28, 28]

    def forward(self, z, labels):
        z = torch.cat((z, labels), dim=1)
        h = self.fc(z)
        h = h.view(-1, 128, 3, 3)
        h = torch.relu(self.deconv1(h))
        h = torch.relu(self.deconv2(h))
        x_reconstructed = torch.sigmoid(self.deconv3(h))
        return x_reconstructed


class CVAE(nn.Module):
    def __init__(self, input_channels, num_classes, latent_dim):
        super(CVAE, self).__init__()
        self.encoder = ConditionalEncoder(input_channels, num_classes, latent_dim)
        self.decoder = ConditionalDecoder(latent_dim, num_classes, input_channels)

    def reparameterize(self, mu, logvar):
        # logvar is log(sigma^2). To obtain sigma use exp(0.5 * logvar)
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
    # labels can be a LongTensor of indices
    if labels.dtype == torch.int64 or labels.dtype == torch.long:
        return torch.eye(num_classes, device=device)[labels]
    else:
        # already one-hot
        return labels.to(device).float()


def train(args):
    # ensure num_gpus default is int
    use_cuda = args.num_gpus > 0
    kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}
    device = torch.device('cuda' if use_cuda else 'cpu')

    logger.info("Iniciando treinamento....")

    train_loader = _get_train_data_loader(args.batch_size, args.data_dir, **kwargs)

    logger.info("Dataset carregado")
    num_epochs = args.epochs
    latent_dim = args.latent_dim
    learning_rate = args.lr
    num_classes = args.num_classes

    model = CVAE(input_channels=1, num_classes=num_classes, latent_dim=latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    logger.info("Modelo Carregado")
    model.train()

    for epoch in tqdm.tqdm(range(num_epochs)):
        train_loss = 0
        for batch_idx, (data, labels) in enumerate(train_loader):
            data = data.to(device)
            labels = one_hot(labels, num_classes, device)

            optimizer.zero_grad()
            x_reconstructed, mu, logvar = model(data, labels)

            assert x_reconstructed.shape == data.shape, f"Shape mismatch: {x_reconstructed.shape} vs {data.shape}"

            loss = loss_function(x_reconstructed, data, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()

        logger.info(f"Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss/len(train_loader.dataset):.4f}")

    model_dir = args.model_dir
    save_model(model, model_dir)


def save_model(model, model_dir):
    logger.info("Saving the model.")
    path = os.path.join(model_dir, "model.pth")
    torch.save(model.cpu().state_dict(), path)
    torch.save(model.decoder.state_dict(), os.path.join(model_dir, 'decoder.pth'))


def _get_train_data_loader(batch_size, training_dir, **kwargs):
    logger.info("Get train data loader")
    dataset = datasets.MNIST(
        training_dir,
        train=True,
        download=True,
        transform=transforms.ToTensor()
    )

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        sampler=None,
        **kwargs
    )


def model_fn(model_dir):
    logger.info("Iniciando a função model_fn para carregar o modelo.")

    latent_dim = int(os.getenv('LATENT_DIM', 120))
    num_classes = int(os.getenv('NUM_CLASSES', 10))
    input_channels = int(os.getenv('INPUT_CHANNELS', 1))

    logger.info("Parâmetros do modelo: Latent Dim = %d, Num Classes = %d, Input Channels = %d",
                latent_dim, num_classes, input_channels)

    model = CVAE(input_channels=input_channels, num_classes=num_classes, latent_dim=latent_dim)
    logger.info("Modelo CVAE inicializado.")

    model_path = os.path.join(model_dir, 'model.pth')
    logger.info("Carregando o modelo do caminho: %s", model_path)

    try:
        state = torch.load(model_path, map_location='cpu')
        model.load_state_dict(state)
        logger.info("Modelo carregado com sucesso.")
    except Exception as e:
        logger.error("Erro ao carregar o modelo: %s", e)
        raise

    model.eval()
    logger.info("Modelo definido para modo de avaliação.")

    return model


def input_fn(request_body, request_content_type):
    logger.info("Recebendo dados com o tipo de conteúdo: %s", request_content_type)

    if request_content_type == 'application/json':
        if isinstance(request_body, str):
            data = json.loads(request_body)
        else:
            data = json.loads(request_body.decode('utf-8'))
        logger.info("Dados recebidos: %s", data)

        if 'label' not in data:
            logger.error("A chave 'label' não está presente no JSON.")
            raise ValueError("JSON must contain a 'label' key with an integer value between 0 and 9.")

        if not isinstance(data['label'], int) or not (0 <= data['label'] <= 9):
            logger.error("O valor da chave 'label' deve ser um inteiro entre 0 e 9: %s", data['label'])
            raise ValueError("JSON must contain a 'label' key with an integer value between 0 and 9.")

        label = torch.tensor([data['label']], dtype=torch.long)
        logger.info("Label convertido em tensor: %s", label)

        return label
    elif request_content_type == 'application/x-npy':
        logger.info("Processando dados de entrada em formato .npy.")
        with io.BytesIO(request_body) as f:
            np_data = np.load(f)
        label = torch.tensor(np_data, dtype=torch.long)
        logger.info("Label convertido de .npy para tensor: %s", label)
        return label
    else:
        logger.error("Tipo de conteúdo não suportado: %s", request_content_type)
        raise ValueError(f"Unsupported content type: {request_content_type}")


def predict_fn(input_data, model):
    logger.info("Iniciando a função de predição com os dados de entrada: %s", input_data)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    latent_dim = int(os.getenv('LATENT_DIM', 120))
    num_classes = int(os.getenv('NUM_CLASSES', 10))

    with torch.no_grad():
        z = torch.randn(1, latent_dim, device=device)
        logger.info("Vetores aleatórios gerados para a predição: %s", z.shape)

        # input_data is expected to be a LongTensor of indices
        if isinstance(input_data, torch.Tensor):
            labels = input_data.to(device)
        else:
            labels = torch.tensor(input_data, dtype=torch.long, device=device)

        labels_one_hot = one_hot(labels, num_classes, device)
        labels_one_hot = labels_one_hot.view(1, -1).float()
        logger.info("Labels one-hot gerados: %s", labels_one_hot.shape)

        model = model.to(device)
        output = model.decoder(z, labels_one_hot)
        logger.info("Saída do modelo obtida com shape: %s", output.shape)

    return output.cpu()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--latent_dim",
        type=int,
        default=120,
        metavar="N",
        help="Latent dim for CVAE (default: 120)",
    )
    parser.add_argument(
        "--batch-size",
        dest='batch_size',
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1000,
        metavar="N",
        help="input batch size for testing (default: 1000)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        metavar="N",
        help="number of epochs to train (default: 10)",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=10,
        metavar="N",
        help="number of classes (default: 10)",
    )
    parser.add_argument(
        "--lr", type=float, default=0.01, metavar="LR", help="learning rate (default: 0.01)"
    )
    parser.add_argument(
        "--momentum", type=float, default=0.5, metavar="M", help="SGD momentum (default: 0.5)"
    )
    parser.add_argument("--seed", type=int, default=1, metavar="S", help="random seed (default: 1)")

    # Container environment
    parser.add_argument("--hosts", type=json.loads, default=os.environ.get("SM_HOSTS", "[]"))
    parser.add_argument("--current-host", type=str, default=os.environ.get("SM_CURRENT_HOST", "localhost"))
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR", "/tmp/model"))
    parser.add_argument("--data-dir", type=str, default=os.environ.get("SM_CHANNEL_TRAINING", "/tmp/data"))
    parser.add_argument("--num-gpus", type=int, default=int(os.environ.get("SM_NUM_GPUS", "0")))

    args = parser.parse_args()
    train(args)
