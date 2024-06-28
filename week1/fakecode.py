from torch.utils.data import Dataset,DataLoader
dataset=Dataset('视频笔记.docx')
dataloader=DataLoader(dataset,batch_size=10,shuffle=True) #训练模式

import torch.nn as nn
import torch.nn.functional as F
class Net(nn.Module)
    def _init_(self):
        super(Net,self)._init_()
        self.layer1=nn.ReLU()
        self.layer2=nn.sigmoid()
        self.layer3=nn.linear()
    def forward(self,x):
        out=self.layer1(x)
        out=self.layer2(out)
        out=self.layer3(out)
        return out
net=Net()
    
import torch.optim as optim #定义Loss函数和优化器
criterion=nn.CrossEntropyLoss()
optimizer=optim.SGD(Net.parameters(),lr=1)
device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

for epoch in range(3): #训练模型
    for x,y in dataloader: #x是输入，y是预期输出
        optimizer.zero_grad()
        x,y=x.to(device),y.to(device) #更换计算单元
        pred=net(x)    #前向传播
        loss=criterion(pred,y) #算loss
        loss.backward() #梯度计算
        optimizer.step() #最优化更新
print('完成训练')
PATH=''
torch.save(net.state_dict(),PATH) #按路径保存模型

##重新加载模型并测试
net=Net()
net.load_state_dict(torch.load(PATH))
dataset=Dataset('视频笔记.docx')
dataloader=DataLoader(dataset,batch_size=10,shuffle=false) #测试模式
net.eval() #验证模式
total_loss=0
for x,y in dataloader:
    x,y=x.to(device),y.to(device)
    with torch.no_grad(): #不算梯度，加速运算
        pred=net(x)
        loss=criterion(pred,y)
    total_loss+=loss.cpu().item()*len(x)
    avg_loss=total_loss/len(dataloader.dataset) #平均loss

correct=0
total=0
with torch.no_grad():
    for data in dataloader:
        x,y=data
        out=net(x)
        _,predicted=torch.max(out.data,1)
        total+=y.size(0)
        correct+=(predicted==y).sum().item()
print('准确率为: {}',format(100*correct/total))
    
