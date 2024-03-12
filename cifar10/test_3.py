import torch
from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms

from src.dataset import Cifar10Dataset
from src.model import LeNet
from src.model import CLASSES, split_dataset

def predict(test_data: Dataset, model: nn.Module) -> None:
    model.eval()
    image = test_data[0][0].unsqueeze(0)
    target = test_data[0][1]
    with torch.no_grad():
        pred = model(image)
        predicted = CLASSES[pred[0].argmax(0)]
        actual = CLASSES[target]
        print((f'Predicted: "{predicted}, Actul: "{actual}"'))
        
def test():
    image_dir = 'data/train'
    test_csv_path = 'data/test_answer.csv'
    
    num_classes = 10
    
    transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5, 0.5) ,(0.5, 0.5, 0.5)
    ])
    
    model = LeNet(num_classes=num_classes)
    model.load_state_dict(torch.load('cifar-net-lenet.pth'))
    
    predict(test_data, model)
    
if __name__=='__main__':
    test()