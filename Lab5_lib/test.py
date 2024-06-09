import torch
from torch.utils.tensorboard.summary import image
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt


# 定义添加椒盐噪声的函数
def add_salt_and_pepper_noise(images, salt_prob=0.05, pepper_prob=0.05):
    """
        添加椒盐噪声到图像张量

        :param images: 输入图像张量，维度为 [batch_size, channels, height, width]
        :param salt_prob: 盐噪声概率
        :param pepper_prob: 椒噪声概率
        :return: 添加噪声后的图像张量
        """
    batch_size, channels, height, width = images.size()

    # 反归一化图像
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(images.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(images.device)
    images = images * std + mean

    # 生成椒盐噪声掩码
    salt_pepper_tensor = torch.rand(batch_size, 1, height, width).to(images.device)
    salt_mask = salt_pepper_tensor < salt_prob
    pepper_mask = salt_pepper_tensor > (1 - pepper_prob)

    # 将掩码扩展到所有通道
    salt_mask = salt_mask.expand(batch_size, channels, height, width)
    pepper_mask = pepper_mask.expand(batch_size, channels, height, width)

    # 应用椒盐噪声
    images[salt_mask] = 1.0
    images[pepper_mask] = 0.0

    # 归一化图像
    images = (images - mean) / std
    return images


def show_image(img_tensor):
    """
    显示图像张量

    :param img_tensor: 输入图像张量，维度为 [channels, height, width]
    """
    # 将图像张量从 [-1, 1] 区间转换到 [0, 1] 区间
    img = img_tensor.cpu()
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img_tensor = img * std + mean
    # 转换为numpy数组，并从 [0, 1] 转换到 [0, 255]
    img_np = img_tensor.numpy().transpose(1, 2, 0)
    img_np = np.clip(img_np, 0, 1)
    img_np = (img_np * 255).astype(np.uint8)

    plt.imshow(img_np)
    plt.axis('off')
    plt.show()

def test(model_address:str  = 'Lab5_lib/resnet50.pth',is_noise:bool=False):
    myWriter = SummaryWriter('Lab5_lib/log/')
    myTransforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    #  load
    test_dataset = torchvision.datasets.CIFAR10(root='Lab5_lib/dataset/', train=False, download=True,
                                                transform=myTransforms)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=True, num_workers=0)

    # 加载模型
    myModel = torch.load(model_address)
    # print(myModel)
    # GPU加速
    myDevice = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    myModel = myModel.to(myDevice)
    correct = 0
    total = 0
    myModel.eval()
    if not is_noise:
        for images, labels in test_loader:
            # GPU加速
            images = images.to(myDevice)
            labels = labels.to(myDevice)
            # show_image(images[0])
            outputs = myModel(images)  # 在非训练的时候是需要加的，没有这句代码，一些网络层的值会发生变动，不会固定
            numbers, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print('Testing Accuracy : %.3f %%' % (100 * correct / total))
        myWriter.add_scalar('test_Accuracy', 100 * correct / total)
    else:
        noise_list = [0.02,0.04,0.06,0.08,0.10]
        for noise in noise_list:
            for images, labels in test_loader:
                # GPU加速
                images = images.to(myDevice)
                labels = labels.to(myDevice)
                images = add_salt_and_pepper_noise(images,noise,noise)
                # show_image(images[0])
                outputs = myModel(images)  # 在非训练的时候是需要加的，没有这句代码，一些网络层的值会发生变动，不会固定
                numbers, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            print('Testing Accuracy : %.3f %%' % (100 * correct / total))
            myWriter.add_scalar('test_Accuracy', 100 * correct / total)


if __name__ == '__main__':
    test(is_noise=True)