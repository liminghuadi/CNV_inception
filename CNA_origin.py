import numpy as np
import pandas as pd
import os
from sklearn.metrics import precision_recall_fscore_support, classification_report, recall_score, precision_score, \
    f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_curve,auc,f1_score
from tensorflow.keras.layers import Activation
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, \
     Convolution1D, MaxPooling1D, Flatten,Conv1D,concatenate,\
     AveragePooling1D,Input,Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical,plot_model

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"   #使用gpu
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # 为使用CPU

data=pd.read_csv(r'D:\a_weka\Weka-3-8\data-my\data.csv',header=0,index_col=0)
data=data.T

target=pd.read_csv(r'D:\a_weka\Weka-3-8\data-my\target.csv',header=None)
#y=target.iloc[:,0].to_list()
y=target.iloc[:,0]
#标签映射为6个数字
y_map={'brca':0,'coadread':1,'gbm':2,'kirc':3,'ov':4,'ucec':5}
y=y.map(y_map)
y=np.array(y)
#y=to_categorical(y,6)

x=np.array(data)
x=x/2

def standardData(data):
    from sklearn.preprocessing import MinMaxScaler
    sts=MinMaxScaler()
    data_std=sts.fit_transform(data)
    return data_std
#x=standardData(x)

data_a=x
target_a=y

#参数列表
seed=42
n_features_bef = len(x[0])
n_features_aft=100
#label_num = np.array(y)
label_num=y
n_classes = 6
k_cross_validation=10
n_features_dec=100

#初始（16，64）
auto_epoch=16
auto_batch_size=64

#初始（32，16）
CNA_epoch=32
CNA_batch_size=16


######## K fold training and test    ########
print("########The results of k fold validation for CNA_origin###########")
print('\n\n\n')
cnn_cvscore = []

def autoencoder_y(X, n_features_bef, n_features_aft, Y):
    input_img = Input(shape=(n_features_bef,))
    encoded = Dense(4096, kernel_initializer='random_uniform', activation='relu')(input_img)
    encoded = Dense(1024, activation='relu')(encoded)
    encoded = Dense(256, activation='relu')(encoded)
    encoded = Dense(n_features_aft, activation='relu')(encoded)

    decoded = Dense(256, activation='relu')(encoded)
    decoded = Dense(1024, activation='relu')(decoded)
    decoded = Dense(4096, activation='relu')(decoded)
    decoded = Dense(n_features_bef, activation='tanh')(decoded)

    autoencoder = Model(inputs=input_img, outputs=decoded)
    encoder = Model(inputs=input_img, outputs=encoded)

    autoencoder.compile(optimizer='adam', loss='mse')
    autoencoder.fit(X, X, epochs=auto_epoch, batch_size=auto_batch_size, shuffle=False)
    X = encoder.predict(X)
    Y = encoder.predict(Y)
    return X, Y
	
def CNA_CNN(x_train, y_train, x_test, y_test, n_features, n_class):
	reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=2, mode='auto')
	input_img = Input(shape=(n_features, ))
	#layer_1 = input_img
	layer_1 = Reshape((n_features, 1))(input_img)
	layer_1 = Conv1D(128, 7, padding='same', activation='relu')(layer_1)
	layer_1 = MaxPooling1D(2, padding='same')(layer_1)
	layer_2 = Conv1D(192, 5, padding='same', activation='relu')(layer_1)
	layer_2 = MaxPooling1D(2, padding='same')(layer_2)
	branch1 = Conv1D(64, 1, padding='same', activation='relu')(layer_2)
	branch2 = Conv1D(48, 1, padding='same', activation='relu')(layer_2)
	branch2 = Conv1D(64, 5, padding='same', activation='relu')(branch2)
	branch3 = Conv1D(48, 1, padding='same', activation='relu')(layer_2)
	branch3 = Conv1D(64, 7, padding='same', activation='relu')(branch3)
	branch4 = Conv1D(48, 1, padding='same', activation='relu')(layer_2)
	branch4 = Conv1D(64, 3, padding='same', activation='relu')(branch4)
	branchpool = MaxPooling1D(3, padding='same', strides=1)(layer_2)
	branchpool = Conv1D(64, 1, padding='same', activation='relu')(branchpool)
	layer_3 = concatenate([branch1, branch2, branch3, branch4, branchpool])
	branch5 = Conv1D(64, 1, padding='same', activation='relu')(layer_3)
	branch6 = Conv1D(48, 1, padding='same', activation='relu')(layer_3)
	branch6 = Conv1D(64, 5, padding='same', activation='relu')(branch6)
	branch7 = Conv1D(48, 1, padding='same', activation='relu')(layer_3)
	branch7 = Conv1D(64, 7, padding='same', activation='relu')(branch7)
	branch8 = Conv1D(48, 1, padding='same', activation='relu')(layer_3)
	branch8 = Conv1D(64, 3, padding='same', activation='relu')(branch8)
	branchpool_2 = MaxPooling1D(3, padding='same', strides=1)(layer_3)
	branchpool_2 = Conv1D(64, 1, padding='same', activation='relu')(branchpool_2)
	layer_4 = concatenate([branch5, branch6, branch7, branch8, branchpool_2])
	layer_5 = AveragePooling1D(3, padding='same')(layer_4)
	layer_5 = Flatten()(layer_5)
	layer_5 = Dropout(0.4)(layer_5)
	layer_6 = Dense(128, activation='relu')(layer_5)
	layer_7 = Dense(n_class, activation='softmax')(layer_6)
	model_classification = Model(inputs=input_img, outputs=layer_7)

	model_classification.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	model_classification.fit(x_train, y_train, epochs=CNA_epoch, CNA_batch_size=16,
							 validation_data=(x_test, y_test),callbacks=[reduce_lr])
	scores = model_classification.evaluate(x_test, y_test)
	y_pred = model_classification.predict(x_test)
	return scores,y_pred

def y_posstoy_pred(y_poss):
    y_pred = []
    for item in y_poss:
        i = np.argmax(item)
        y_pred.append(i)
    return np.array(y_pred)


cnn_pre_total = np.zeros(n_classes)
cnn_rec_total = np.zeros(n_classes)
cnn_fsc_total = np.zeros(n_classes)

from sklearn.model_selection import StratifiedKFold,KFold
skf=StratifiedKFold(n_splits=k_cross_validation,shuffle=True,random_state=seed)
for train_index,test_index in skf.split(data_a,target_a):
    data_trainvalid,target_trainvalid=data_a[train_index],target_a[train_index]
    data_test,target_test=data_a[test_index],target_a[test_index]

    data_trainvalid=pd.DataFrame(data_trainvalid)#索引为乱序
    target_trainvalid=pd.DataFrame(target_trainvalid)#索引为乱序
    #用于测试分类模型
    x_test=np.array(pd.DataFrame(data_test))
    y_test=np.array(pd.DataFrame(target_test))

    # 训练集索引重新排序
    x_train = np.array(data_trainvalid.reset_index(drop=True))
    y_train = np.array(target_trainvalid.reset_index(drop=True))

    n_features=x_train.shape[1]
    x_train_d,x_test_d = autoencoder_y(x_train,n_features,n_features_dec,x_test)
    scores,y_poss=CNA_CNN(x_train_d,y_train,x_test_d,y_test,n_features_dec,n_classes)

    y_pred = y_posstoy_pred(y_poss)
    cnn_cvscore.append(scores[1])

    cnn_precision, cnn_recall, cnn_fscore, _ = precision_recall_fscore_support(y_test, y_pred)
    cnn_pre_total = cnn_pre_total + cnn_precision
    cnn_rec_total = cnn_rec_total + cnn_recall
    cnn_fsc_total = cnn_fsc_total + cnn_fscore

cnn_cvscore=np.array(cnn_cvscore)
cnn_pre_total = cnn_pre_total / k_cross_validation
cnn_rec_total = cnn_rec_total / k_cross_validation
cnn_fsc_total = cnn_fsc_total / k_cross_validation

print('k-fold:',k_cross_validation)
print('The precision of classification is :', np.round(cnn_pre_total,4))
print('The recall of classification is :', np.round(cnn_rec_total,4))
print('The fscore of classification is :', np.round(cnn_fsc_total,4))
print('The accuracy of classification of k fold validation is:', np.round(cnn_cvscore,4))

record={'precision':np.round(cnn_pre_total, 4),
        'recall':np.round(cnn_rec_total, 4),
        'fscore':np.round(cnn_fsc_total, 4)
        }
rec=pd.DataFrame(record)
acc=pd.DataFrame(np.round(cnn_cvscore,4),columns=['acc'])
rec=pd.concat([rec,acc],axis=1)
rec.to_csv(r'C:\Users\wanqikang\Desktop\CNV实验\CNA_origin_result.csv')

