from transformers import AutoImageProcessor, ResNetForImageClassification
import torch
from torchvision import datasets,transforms
from torch.utils.data import DataLoader

image_processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")
model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features,10)
model.eval()

transform=transforms.Compose([
    transforms.Resize((224,224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor()
])

test = datasets.MNIST(root="./test",train=False,transform=transform, download=True)
dataloader = DataLoader(test, batch_size=128)

correct = 0
total = 0

with torch.no_grad():
    for image, label in dataloader:
        output = model(image).logits
        _,predicted = torch.max(output, 1)
        total += label.size(0)
        correct += (predicted[:label.size(0)] == label).sum().item()
acc = correct / total
print(f' {acc*100:.2f}%')
