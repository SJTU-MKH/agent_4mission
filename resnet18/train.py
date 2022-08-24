import os
import time
import torch.nn as nn
import torch
import torchvision.transforms as transforms
from PIL import Image
from matplotlib import pyplot as plt
import torch
import torchvision.models as models
from torch.utils.data import  DataLoader
import torchsummary
from torch.autograd import Variable
import copy
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

os.environ['CUDA_VISIBLE_DEVICES']='1'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MyDataset(torch.utils.data.Dataset):
    def __init__(self,file_path,transform=None):
        super(MyDataset,self).__init__()
        #依次处理数据
        imgs = []
        imglb = []
        classes = list(range(8))
        for i in classes:
            img_names = os.listdir(os.path.join(file_path,str(i)))
            for j in range(len(img_names)):
                path = os.path.join(file_path, str(i), img_names[j])
                label = i

                #读对应路径的图像，存为img，依次存入imgs
                img = Image.open(path).convert('RGB')
                imgs.append(img)
                #读对应图像的标签，依次存入imglb  
                imglb.append(int(label))
        self.image = imgs
        self.imglb = imglb
        self.root = file_path
        self.size = len(imgs)
        self.transform = transform
        
    def __getitem__(self,index):
        img = self.image[index]
        label = self.imglb[index]
        sample = {'image': img,'classes':label}
        if self.transform:
            sample['image'] = self.transform(img)
        return sample
        
    def __len__(self):
        return self.size

norm_mean = [0.485, 0.456, 0.406]
norm_std = [0.229, 0.224, 0.225]

train_transforms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(p=0.2),
    transforms.RandomVerticalFlip(p=0.2),
    transforms.RandomRotation(degrees=(-10,10)),
    # transforms.ColorJitter(brightness=0.1),
    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std),
])

val_transforms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std),
])



trainpath = '/home3/HWGroup/liujy/agent_4mission_detection/resnet18/datasets_use/train'
valpath = '/home3/HWGroup/liujy/agent_4mission_detection/resnet18/datasets_use/test'
CLASSES = ['red', 'green', 'blue', 'yellow', 'pin', 'qing', 'black', 'white']
train_dataset = MyDataset(file_path = trainpath, transform = train_transforms)
test_dataset = MyDataset(file_path = valpath, transform = val_transforms)
print(train_dataset.size)
print('=========')
#设置batch size = 16
train_loader = DataLoader(dataset=train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(dataset=test_dataset)
data_loaders = {'train': train_loader, 'val': test_loader}

def get_model(m_path=None, vis_model=False):

    resnet18 = models.resnet18(pretrained=False)
    # torchsummary.summary(resnet18, (3,224,224))
    # 修改全连接层的输出
    num_ftrs = resnet18.fc.in_features
    resnet18.fc = nn.Linear(num_ftrs, 8)

    # 加载模型参数
    if m_path:
        checkpoint = torch.load(m_path)
        resnet18.load_state_dict(checkpoint['model_state_dict'])


    if vis_model:
        from torchsummary import summary
        summary(resnet18, input_size=(3, 224, 224), device=device)

    return resnet18

net = get_model()
net = net.to(device)


def train_model(model, criterion, optimizer, scheduler, num_epochs=50):
    Loss_list = {'train': [], 'val': []}
    Accuracy_list_species = {'train': [], 'val': []}

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        print('epoch {}/{}'.format(epoch,num_epochs - 1))
        print('-*' * 10)
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            corrects_species = 0
            
            for idx,data in enumerate(data_loaders[phase]):
                #print(phase+' processing: {}th batch.'.format(idx))
                inputs = Variable(data['image'].cuda())
                labels_species = Variable(data['classes'].cuda())
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    x_species = model(inputs)
        
                    _, preds_species = torch.max(x_species, 1)

                    loss = criterion(x_species, labels_species)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                running_loss += loss.item() * inputs.size(0)

                corrects_species += torch.sum(preds_species == labels_species)
            
            epoch_loss = running_loss / len(data_loaders[phase].dataset)
            Loss_list[phase].append(epoch_loss)

            epoch_acc_species = corrects_species.double() / len(data_loaders[phase].dataset)
            epoch_acc = epoch_acc_species

            Accuracy_list_species[phase].append(100 * epoch_acc_species)
            print('{} Loss: {:.4f}  Acc_species: {:.2%}'.format(phase, epoch_loss,epoch_acc_species))

            if phase == 'val' and epoch_acc > best_acc:

                best_acc = epoch_acc_species
                best_model_wts = copy.deepcopy(model.state_dict())
                print('Best val species Acc: {:.2%}'.format(best_acc))

    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), f'{savepath}best_model.pth')
    # torch.save(model.state_dict(), 'best_model.pth')
    print('Best val species Acc: {:.2%}'.format(best_acc))
    return model, Loss_list,Accuracy_list_species

savepath = './runs/model2.0/'
if not os.path.isdir(savepath):
    os.mkdir(savepath)

network = net.cuda()
optimizer = torch.optim.SGD(network.parameters(), lr=0.01, momentum=0.9)
criterion = nn.CrossEntropyLoss()
exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1) # Decay LR by a factor of 0.1 every 1 epochs

print("===========start training=============")
model, Loss_list, Accuracy_list_species = train_model(network, criterion, optimizer, exp_lr_scheduler, num_epochs=200)


import matplotlib.pyplot as plt
x = range(0, 200)
y1 = Loss_list["val"]
y2 = Loss_list["train"]
plt.figure(figsize=(18,14))
plt.subplot(211)
plt.plot(x, y1, color="r", linestyle="-", marker="o", linewidth=1, label="val")
plt.plot(x, y2, color="b", linestyle="-", marker="o", linewidth=1, label="train")
plt.legend()
plt.title('train and val loss vs. epoches')
plt.ylabel('loss')
plt.savefig(f'{savepath}train and val loss vs epoches.jpg')

plt.subplot(212)
y5 = Accuracy_list_species["train"]
y6 = Accuracy_list_species["val"]
plt.plot(x, y5, color="r", linestyle="-", marker=".", linewidth=1, label="train")
plt.plot(x, y6, color="b", linestyle="-", marker=".", linewidth=1, label="val")
plt.legend()
plt.title('train and val Species_acc vs. epoches')
plt.ylabel('species_accuracy')
plt.savefig(f'{savepath}train and val Classes_acc vs epoches.jpg')