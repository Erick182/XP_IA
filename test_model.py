import torch
import matplotlib.pyplot as plt
from training_script_fixed import CVAE, one_hot

# ======= CONFIGURAÇÃO =======
model_path = "model/model.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======= INICIALIZAÇÃO DO MODELO =======
model = CVAE(input_channels=1, num_classes=10, latent_dim=120).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# ======= GERAR AMOSTRAS =======
num_samples = 16
labels = torch.randint(0, 10, (num_samples,))
labels_one_hot = one_hot(labels, num_classes=10, device=device)

with torch.no_grad():
    z = torch.randn(num_samples, 120).to(device)
    samples = model.decoder(z, labels_one_hot)

# ======= PLOTAR E SALVAR AMOSTRAS =======
fig, axes = plt.subplots(4, 4, figsize=(6,6))
for i, ax in enumerate(axes.flatten()):
    ax.imshow(samples[i].cpu().reshape(28,28), cmap="gray")
    ax.axis("off")

plt.tight_layout()
plt.savefig("sample_generated.png")
print("Imagem gerada: sample_generated.png")
