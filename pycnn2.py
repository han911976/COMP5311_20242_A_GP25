
import os
import torch
import torch.nn as nn
import torch.cuda
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
from torchsummary import summary
from matplotlib import cm
from sklearn.manifold import TSNE
# from keras.utils import np_utils
import numpy as np
from sklearn.utils import shuffle
from pandas import DataFrame
# from conf_plt import *
from sklearn.metrics import confusion_matrix, accuracy_score
import time


# Hyper Parameters
EPOCH = 10 # 30	   # train the training data n times, to save time, we just train 1 epoch
BATCH_SIZE = 256
LR = 0.0001			  # learning rate
DOWNLOAD_MNIST = False
LOG_FILE = 'finally_log.txt'

RANDOM_STATE=5311
torch.manual_seed(RANDOM_STATE)

# arg_list = [[40,3],[40,5],[80,3],[80,5]]
arg_list = [[10,2],[50,2],[50,3],[50,4], [80,2], [80,5]]

dict_14class = {0:'Outlook',1:'Facetime',2:'Skype',3:'SMB',4:'Gmail',5:'Weibo',6:'FTP'
				,7:'WorldOfWarcraft',8:'MySQL',9:'BitTorrent',10:'http',11:'syn',12:'udp',13:'ack'}
dict_2class = {0:'Benign', 1:'Malware'}
dict_20class = {0:'Outlook',1:'Facetime',2:'Skype',3:'SMB',4:'Gmail',5:'Weibo',6:'FTP'
				,7:'WorldOfWarcraft',8:'MySQL',9:'BitTorrent',10:'Miuref',11:'Shifu',12:'Tinba'
				,13:'Nsis-ay',14:'Neris',15:'Zeus',16:'Cridex',17:'Geodo',18:'Htbot',19:'Virut'}


benign_m = 0
attack_m = 0
#b_size = 2400
#m_size = 5990
b_size = 5997
m_size = 5997
NUM_classes=10

# dict_20class = [dict_20class[i] for i in range(20)]
# plot_14columns = [dict_14class[i] for i in range(14)]
# plot_2columns = [dict_2class[i] for i in range(2)]
# plot_10columns = [dict_20class[i] for i in range(10)]
testtime=[]


class Session:

	def __init__(self, pshape, Benign_all_loss, Benign_loss, Malware_loss, Benign_mean, Benign_sd, Benign_max, CNN_confm, auto_confm, CNN_accuracy, auto_accuracy):
		self.pshape = pshape
		self.Benign_all_loss = Benign_all_loss
		self.Benign_loss = Benign_loss
		self.Malware_loss = Malware_loss
		self.Benign_mean = Benign_mean
		self.Benign_sd = Benign_sd
		self.Benign_max = Benign_max
		self.CNN_confm = CNN_confm
		self.auto_confm = auto_confm
		self.CNN_accuracy = CNN_accuracy
		self.auto_accuracy = auto_accuracy
	def show_header(self):
		print(" "*8+"{:15}".format('pshape')
				+"{:15}".format('Benign_Mean')+"{:15}".format('Benign_max')
				+"{:15}".format('CNN_accuracy')+"{:15}".format('auto_accuracy'))
	def show_contain(self):
		for x,y in self.__dict__.items() :
			if x not in  ['CNN_confm','auto_confm','Benign_loss','Benign_all_loss','Malware_loss','Benign_sd']:
#				print(type(y),x)
				if type(y) is np.float64:
					print("{:15}".format(format(y,'.6f')),end='')
				else :
					print("{:15}".format(str(y)),end='')
	def show_contain2(self):
		for x,y in self.__dict__.items() :
			if x not in  ['CNN_confm','auto_confm','Benign_loss','Benign_all_loss','Malware_loss','Benign_sd']:
#				print(type(y),x)
				if type(y) is np.float64:
					return "{:15}".format(format(y,'.6f'))
				else :
					return "{:15}".format(str(y))

def input_Data(Data_path, pshape, size):
	global img_shape
	global test_sample
	global arg_list
	global label
	global testtime
	Data=[]

	for flows in os.listdir(Data_path):
		tmp_read = np.load(Data_path+flows, allow_pickle=True)
		print(flows)
		test_start = time.process_time()
		for i, flow in enumerate(tmp_read):
			if i >= size:
				break
			img_Data=[]
			for j, pkt in enumerate(flow):
				if j >= pshape[1]:
					break
				tmp_=[]
				for k, value in enumerate(pkt):
					if k >= pshape[0]:
						break
					tmp_.append(value)
				if len(tmp_) < pshape[0]:
					tmp_.extend([0]*(pshape[0] - len(tmp_)))

				img_Data.extend(tmp_)
			img_Data=np.asarray(img_Data)
			img_Data.resize(img_shape[0],img_shape[1])
			Data.append(img_Data)
		print(len(Data))
		print(len(Data) / (time.process_time() - test_start) ,'flows/second')
		testtime.append(len(Data) / (time.process_time() - test_start))
	return Data

def plot_with_labels(lowDWeights, labels):
	plt.cla()
	X, Y = lowDWeights[:, 0], lowDWeights[:, 1]
	for x, y, s in zip(X, Y, labels):
		c = cm.rainbow(int(255 * s / 9)); plt.text(x, y, s, backgroundcolor=c, fontsize=9)
	plt.xlim(X.min(), X.max()); plt.ylim(Y.min(), Y.max()); plt.title('Visualize Dense layer(1024 dim)'); plt.savefig('foo.png'); plt.pause(0.01)

def batch(iterable1,iterable2, n=1):
	if len(iterable1) != len(iterable2):
		raise Exception('The Data and Label size error')
	l = len(iterable1)
	for ndx in range(0, l, n):
		yield iterable1[ndx:min(ndx + n, l)], iterable2[ndx:min(ndx + n, l)]

class CNN_AUTO(nn.Module):
	def __init__(self,input_size):
		super(CNN_AUTO, self).__init__()
		self.conv1 = nn.Sequential(		 # input shape (1, 28, 28)
			nn.Conv1d(
				in_channels=1,			  # input height
				out_channels=32,			# n_filters
				kernel_size=6,			  # filter size
				stride=1,				   # filter movement/step
				padding=5,				  # if want same width and length of this image after Conv2d, padding=(kernel_size-1)/2 if stride=1
			),							  # output shape (16, 28, 28)
			nn.ReLU(),					  # activation
			nn.MaxPool1d(kernel_size=2),	# choose max value in 2x2 area, output shape (16, 14, 14)
		)
		input_size = (1,32,int(((input_size[1]-6+5*2)/1+1)/2))
		self.conv2 = nn.Sequential(		 # input shape (16, 14, 14)
			nn.Conv1d(32, 64, 6, 1, 5),	 # output shape (32, 14, 14)
			nn.ReLU(),					  # activation
			nn.MaxPool1d(kernel_size=2),				# output shape (32, 7, 7)
		)
		input_size = (1,64,int(((input_size[2]-6+5*2)/1+1)/2))
#		self,flatten = nn.Linear(np.prod(input_size[1:]), 10)
		self.dense1 = nn.Linear(np.prod(input_size[1:]), 1024)
#		self,dense1_1 = nn.Linear(1024, 10)
		self.dense2 = nn.Linear(1024, 25)
		self.cnn_out = nn.Linear(25, 10)	# fully connected layer, output 10 classes
		self.encoder_1 = nn.Linear(1024, 512)
		self.encoder_2 = nn.Linear(512, 256)
		self.decoder_1 = nn.Linear(256, 512)
		self.decoder_2 = nn.Linear(512, 1024)


	def forward(self, x):
		x = self.conv1(x)
		x = self.conv2(x)
		x = x.view(x.size(0), -1)	   # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
		mid = self.dense1(x)
		x = self.dense2(mid)
		cnn_output = self.cnn_out(x)
		x = self.encoder_1(mid)
		x = self.encoder_2(x)
		x = self.decoder_1(x)
		x = self.decoder_2(x)
		return cnn_output, mid, x  # return mid for visualization

if __name__ == "__main__":

	DATAS = []

	for pshape in arg_list:

		img_shape = (1,pshape[0]*pshape[1])

		Data = []
		Label = []

		try:

			del X_train
			del y_train
			del X_test
			del B_test
			del M_test
			del B_test_test
			del M_test_test
			del b_x
			del b_y
		except Exception:
			pass

		B_data = input_Data('data/USTC_benign/', pshape, b_size)
		M_data = input_Data('data/USTC_attack/', pshape, m_size)

		benign_m = len(B_data)
		attack_m = len(M_data)
		L=0

		for i in range(10):

			Label.extend([L]*int(b_size))
			L=L+1

		for i in range(4):
			Label.extend([L]*int(m_size))
			L=L+1


		Data.extend(B_data)
		Data.extend(M_data)
		Data = np.asarray(Data)
		B_data = np.asarray(B_data)
		M_data = np.asarray(M_data)
		Label = np.asarray(Label)

		X_train = np.delete(Data[:59970],np.s_[:59970:9],0).reshape(-1,img_shape[0],img_shape[1]).astype('float32') / 255
		y_train = np.delete(Label[:59970],np.s_[:59970:9],0)
#
		X_test = Data[:59970:9].reshape(-1,img_shape[0],img_shape[1]).astype('float32') / 255
		y_test = Label[:59970:9]

		B_data = shuffle(B_data, random_state=RANDOM_STATE)
		B_test = B_data[:23988].reshape(-1,img_shape[0],img_shape[1]).astype('float32') / 255
		B_test_test = B_test[::9]
		B_test = np.delete(B_test,np.s_[::9],0)
#		B_label = np_utils.to_categorical(Label[:59970], num_classes=NUM_classes)
#
		M_test = M_data.reshape(-1,img_shape[0],img_shape[1]).astype('float32') / 255
		M_test_test = M_test[::9]
		M_test = np.delete(M_test,np.s_[::9],0)
#		M_label = np_utils.to_categorical(Label[59970:], num_classes=NUM_classes)

		X_train, y_train = shuffle(X_train, y_train, random_state=RANDOM_STATE)
		X_test, y_test = shuffle(X_test, y_test, random_state=RANDOM_STATE)

		X_train = torch.from_numpy(X_train) # .cuda()
		y_train = torch.from_numpy(y_train) # .cuda()
		X_test = torch.from_numpy(X_test) # .cuda()
#		y_test = torch.from_numpy(y_test).cuda()


		B_test = torch.from_numpy(B_test) # .cuda()
		M_test = torch.from_numpy(M_test) # .cuda()
		B_test_test = torch.from_numpy(B_test_test) # .cuda()
		M_test_test = torch.from_numpy(M_test_test) # .cuda()

		print("\n===========================\n")
		print("Train_packet \t packet size : "+str(pshape[0])+"\t\tpacket count : "+str(pshape[1]))
		print("Classify to {} class (Deep_Benign vs. Four_Mirai)".format(NUM_classes))
		print("Train_shape \t: "+str(img_shape))
		print("Train sample \t: "+str(len(y_train)))
		print("Test sample \t: "+str(len(y_test)))
		print("\n===========================\n")

		cnn_auto = CNN_AUTO([img_shape[0],img_shape[1]])
		print(cnn_auto)  # net architecture
#		summary(cnn,(1,80))
		cnn_auto # .cuda()

		optimizer = torch.optim.Adam(cnn_auto.parameters(), lr=LR)   # optimize all cnn parameters

		# training and testing
		for epoch in range(EPOCH):
			# gives batch data, normalize x when iterate train_loader
			epoch_start = time.time()
			for step, (b_x, b_y) in enumerate(batch(X_train,y_train,BATCH_SIZE)):
				b_x = b_x # .cuda()
				b_y = b_y # .cuda()
				cnn_output = cnn_auto(b_x)[0]			   # cnn_auto output
				auto_input = cnn_auto(b_x)[1]
				auto_output = cnn_auto(b_x)[2]
				loss = nn.CrossEntropyLoss()(cnn_output, b_y) + nn.MSELoss()(auto_input, auto_output)  # cross entropy loss
				optimizer.zero_grad()		   # clear gradients for this training step
				loss.backward()				 # backpropagation, compute gradients
				optimizer.step()				# apply gradients

			print('Epoch: ', epoch, '\t| train loss: %.4f' % loss.data.item(), '\t time %.4f' % (time.time() - epoch_start))

		tcnn_out,_,_ = cnn_auto(X_test)
		tcnn_out = tcnn_out.cpu()
		pred_y = torch.max(tcnn_out, 1)[1].data.numpy()
		CNN_confm = confusion_matrix(y_test, pred_y)
		_,b_a_in,b_a_out = cnn_auto(B_test)
		_,m_a_in,m_a_out = cnn_auto(M_test)

		B_loss = np.array([nn.MSELoss()(b_a_in[i], b_a_out[i]).item() for i in range(len(B_test))])
		M_loss = np.array([nn.MSELoss()(m_a_in[i], m_a_out[i]).item() for i in range(len(M_test))])

		B_SD = np.std(B_loss,ddof=1)
		B_mean = B_loss.sum()/len(B_loss)
		B_max = B_loss.max()
		B_all_loss = B_loss

		Acc=[]

		_,b_a_in,b_a_out = cnn_auto(B_test_test)
		_,m_a_in,m_a_out = cnn_auto(M_test_test)

		B_loss = np.array([nn.MSELoss()(b_a_in[i], b_a_out[i]).item() for i in range(len(B_test_test))])
		M_loss = np.array([nn.MSELoss()(m_a_in[i], m_a_out[i]).item() for i in range(len(M_test_test))])

		TP = len(B_loss[B_loss <= B_max])
		FN = len(B_loss) - TP
		FP = len(M_loss[M_loss <= B_max])
		TN = len(M_loss) - FP

		auto_confm = np.array([[TP,FN],[FP,TN]])

		DATAS.append(Session(pshape,B_all_loss,B_loss,M_loss,B_mean,B_SD,B_max,CNN_confm,auto_confm,CNN_confm.diagonal().sum()/CNN_confm.sum(),auto_confm.diagonal().sum()/auto_confm.sum()))

	DATAS[0].show_header()
	for i in range(len(DATAS)):
		print("{}".format("TEST"+str(i)+"\t"),end='')
		DATAS[i].show_contain()
		print()

	for i in	 range(len(DATAS)):
		B_max=np.asarray(sorted(DATAS[i].Benign_all_loss)[-len(DATAS[i].Benign_all_loss)//100:]).mean()
		B_loss=DATAS[0].Benign_loss

		TP = len(B_loss[B_loss <= B_max])
		FN = len(B_loss) - TP
		FP = len(M_loss[M_loss <= B_max])
		TN = len(M_loss) - FP

		auto_confm = np.array([[TP,FN],[FP,TN]])
		DATAS[i].auto_accuracy = auto_confm.diagonal().sum()/auto_confm.sum()
		DATAS[i].Benign_max = B_max

	DATAS[0].show_header()
	for i in range(len(DATAS)):
		print("{}".format("TEST"+str(i)+"\t"),end='')
		DATAS[i].show_contain()
		print()

try:
	with open(LOG_FILE,'a') as f:
		f.write("\n")
		f.write(" "*8+"{:15}".format('pshape')
					+"{:15}".format('Benign_Mean')+"{:15}".format('Benign_max')
					+"{:15}".format('CNN_accuracy')+"{:15}".format('auto_accuracy'))
		t = time.strftime('%Y-%m-%d %X',time.localtime())
		f.write(t + "\n")
		for i in range(len(DATAS)):
			f.write("{}".format("TEST"+str(i)+"\t"))
			f.write(DATAS[i].show_contain2())
			f.write("\n")
except:
	print("Fail to write log")

print("END.")
