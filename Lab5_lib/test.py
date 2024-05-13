import torch
from torch.utils.tensorboard.summary import image
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

myWriter = SummaryWriter('log/')

myTransforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

#  load
test_dataset = torchvision.datasets.CIFAR10(root='./dataset/', train=False, download=True,
                                            transform=myTransforms)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=True, num_workers=0)

# 加载模型
myModel = torch.load('resnet50.pth')
print(myModel)
# GPU加速
myDevice = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
myModel = myModel.to(myDevice)
correct = 0
total = 0
myModel.eval()
for images, labels in test_loader:
    # GPU加速
    images = images.to(myDevice)
    labels = labels.to(myDevice)
    outputs = myModel(images)  # 在非训练的时候是需要加的，没有这句代码，一些网络层的值会发生变动，不会固定
    numbers, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()
print('Testing Accuracy : %.3f %%' % (100 * correct / total))
myWriter.add_scalar('test_Accuracy', 100 * correct / total)