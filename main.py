
import random
import math
class bp:
    def init_w(self,w,x,y):#初始化偏置值
        for i in range(x):
            for j in range(y):
                w[i][j]=random.random()
                if(w[i][j]<0.5):w[i][j]=-w[i][j];
    def init_se(self,w,x):#初始化权值
        for i in range(x):
            w[i]=random.random()
            if(w[i]<0.5):
                w[i]=-w[i]

    def  forward(self,   inp,  outp,w, x, y, se):#向前传播输入
        for j in range(y):
            outp[j]=0
            for i in range(x):
                outp[j]+=inp[i]*w[i][j]
            outp[j] = outp[j]+se[j];
            outp[j] = (1.0)/(1+math.exp(-outp[j]))
    def  reforward(self):#反向误差更新
        self.sumse = 0;
        #计算输出层误差
        for i in range(self.o_size):
            self.eo[i] = self.ouput[i] * (1.0-self.ouput[i]) * (self.ouputex[i]-self.ouput[i]);
            if(self.eo[i]<0.0):
                self.sumse -= self.eo[i];
            else:
                self.sumse += self.eo[i]
        #计算输入层误差
        for i in range(self.h_size):
            self. eh[i] = 0;
            for j in range(self.o_size):
                self. eh[i]+= self.hidden[i] * (1-self.hidden[i]) * self.who[i][j] * self.eo[j];
    def  updatew(self):
        #更新隐含层与输出层权值
        for i in range(self.h_size):
            for j in range(self.o_size):
                self.upwho[i][j]=(self.L*self.hidden[i]*self.eo[j])+(self.Mom*self.upwho[i][j])
                self.who[i][j]+=self.upwho[i][j]
        #更新输入与隐含层权值
        for i in range(self.i_size):
            for j in range(self.h_size):
                self.upwih[i][j]=(self.L*self.input[i]*self.eh[j])+(self.Mom*self.upwih[i][j])
                self.wih[i][j]+=self.upwih[i][j]
                #更新阈值
    def  updatefa(self):
        for i in range(self.i_size):
            self.seh[i]+=self.L*self.eh[i]
        for i in range(self.o_size):
            self.seo[i]+=self.L*self.eo[i]
        #训练函数
    def  train(self ,in1,out1):
        self.input=in1
        self.ouputex=out1
        self.forward(self.input, self.hidden, self.wih, self.i_size, self.h_size, self.seh)#向前传播输入
        self.forward(self.hidden, self.ouput, self.who, self.h_size, self.o_size, self.seo)
        self.reforward()#反向误差传播
        self.updatew()#更新网络权重
        self.updatefa()#更新阈值
    #测试函数
    def test(self,init1):
        self.input=init1
        self.forward(self.input, self.hidden, self.wih, self.i_size, self.h_size, self.seh)
        self.forward(self.hidden, self.ouput, self.who, self.h_size, self.o_size, self.seo)
        for i in range(self.o_size):
            print(self.ouput[i])
    #返回阈值
    def get_e(self,w,x):
        self.f=0
        for i in range(self.o_size):
            print(w[i],end="")
            self.f+=1
            if(self.f%2==0):
                print("")
        print("")
    #返回权值
    def get_w(self,w,x,y):
        self.f=0;
        for i in range(x):
            for j in range(y):
                print(w[i][j],end="")
                print("     ",end="")
                self.f+=1
                if(self.f%2==0):
                    print("")
                    #类初始化
    def __init__(self,size,l,mom):

        self.L=l                        #学习因子
        self.Mom=mom;        #动量
        self.i_size=size[0];          #输入层数量
        self.h_size=size[1];        #隐含层数量
        self.o_size=size[2];          #输出层数量
        self.wih=[[0 for i in range(self.h_size)] for j in range(self.i_size)]    #输入层与隐含层权值
        self.who=[[0 for i in range(self.o_size)] for j in range(self.h_size)]     #隐含层与输出层权值
        self.upwih=[[0 for i in range(self.h_size)] for j in range(self.i_size)]
        self.upwho=[[0 for i in range(self.o_size)] for j in range(self.h_size)]     # 动量更新
        self.input=[0 for i in range(self.i_size)]                      #输入层
        self.hidden=[0 for i in range(self.h_size)]                #隐含层
        self.ouput=[0 for i in range(self.o_size)]                   #输出层
        self.ouputex=[0 for i in range(self.o_size)]                #期待输出
        self.seh= [0 for i in range(self.h_size)]                       #隐含层偏置
        self.seo= [0 for i in range(self.o_size)]                       #输出层偏置
        self.eh= [0 for i in range(self.h_size)]                       #隐含层误差
        self.eo= [0 for i in range(self.o_size)]                       #输出层误差

        self. init_w(self.wih,self.i_size,self.h_size)          #初始化 输入层到隐含层
        self. init_w(self.who,self.h_size,self.o_size)         #初始化 隐含层到输出层.

        self.init_se(self.seh, self.h_size)#初始化隐含层的偏置
        self.init_se(self.seo, self.o_size)#初始化输出层的偏置


for i in range(10000):
    for j in range(3):
        t.train(inputData[j],outputData[j])
    if(t.sumse<0.0001):
        print("迭代次数:",i)
        break

print("误差为:",t.sumse)
print("输入层与隐含层连接权值为:",end="")
t.get_w(t.wih,t.i_size,t.h_size)
print("隐含层与输出层连接权值为:",end="")
t.get_w(t.who,t.h_size,t.o_size)
print("隐含层神经元阈值为:",end="")
t.get_e(t.seh, t.h_size)
print("输出层神经元阈值为:",end="")
t.get_e(t.seo, t.o_size)

for i in range(3):
    print("训练样本为:",testData[i],"        结果为：",end="")
    t.test(testData[i])


size =[2,2,1]
inputData = [[0.0,0.0],[0.0,1.0],[1.0,0.0],[1.0,1.0],[0.1,1.0],[0.1,1.0],[0.1,1.0]]
outputData = [[0.0],[1.0],[1.0],[0.0],[1.0],[1.0],[1.0]]
testData = [[0.05,0.1],[0.2,0.9],[0.86,0.95]]


t=bp(size,0.6,0.5)

