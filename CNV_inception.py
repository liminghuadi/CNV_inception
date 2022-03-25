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
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"   #使用gpu
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
	
#inception
def inception(x, nb_filter1, nb_filter2):
    branch1 = Conv1D(nb_filter2, 1, padding='same', activation='relu')(x)

    branch3 = Conv1D(nb_filter1, 1, padding='same', activation='relu')(x)
    branch3 = Conv1D(nb_filter2, 3, padding='same', activation='relu')(branch3)

    branch5 = Conv1D(nb_filter1, 1, padding='same', activation='relu')(x)
    branch5 = Conv1D(nb_filter2, 3, padding='same', activation='relu')(branch5)
    branch5 = Conv1D(nb_filter2, 3, padding='same', activation='relu')(branch5)

    branch7 = Conv1D(nb_filter1, 1, padding='same', activation='relu')(x)
    branch7 = Conv1D(nb_filter2, 3, padding='same', activation='relu')(branch7)
    branch7 = Conv1D(nb_filter2, 3, padding='same', activation='relu')(branch7)
    branch7 = Conv1D(nb_filter2, 3, padding='same', activation='relu')(branch7)

    branchpool = MaxPooling1D(3, padding='same', strides=1)(x)
    branchpool = Conv1D(nb_filter2, 1, padding='same', activation='relu')(branchpool)

    x = concatenate([branch1, branch3, branch5, branch7, branchpool])

    return x

def CNV_inception(x_train, y_train, x_test, y_test, n_features, n_class):
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=2, mode='auto')
    input_img = Input(shape=(n_features,))
    x = Reshape((n_features, 1))(input_img)
    x = Conv1D(128, 7, padding='same', activation='relu')(x)
    x = MaxPooling1D(2, padding='same')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.12)(x)
    x = Conv1D(192, 5, padding='same', activation='relu')(x)
    x = MaxPooling1D(2, padding='same')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.12)(x)
    x = inception(x, 48, 64)
    x = inception(x, 48, 64)
    x = inception(x, 48, 64)
    x = AveragePooling1D(3, padding='same')(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    output_img = Dense(n_class, activation='softmax')(x)
    model = Model(inputs=input_img, outputs=output_img)
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam',
                  metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=CNA_epoch, batch_size=CNA_batch_size,
              validation_data=(x_test, y_test), callbacks=[reduce_lr])
    scores = model.evaluate(x_test, y_test)
    y_pred = model.predict(x_test)
    return scores, y_pred

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
    scores,y_poss=CNV_inception(x_train_d,y_train,x_test_d,y_test,n_features_dec,n_classes)

    y_pred = y_posstoy_pred(y_poss)
    cnn_cvscore.append(scores[1])

    cnn_precision, cnn_recall, cnn_fscore, _ = precision_recall_fscore_support(y_test, y_pred)
    cnn_pre_total = cnn_pre_total + cnn_precision
    cnn_rec_total = cnn_rec_total + cnn_recall
    cnn_fsc_total = cnn_fsc_total + cnn_fscore
    # print('The precision of one time cnn classification is :', cnn_precision)
    # print('The recall of one time cnn classification is :', cnn_recall)
    # print('The fscore of one time cnn classification is :', cnn_fscore)

cnn_cvscore=np.array(cnn_cvscore)
cnn_pre_total = cnn_pre_total / k_cross_validation
cnn_rec_total = cnn_rec_total / k_cross_validation
cnn_fsc_total = cnn_fsc_total / k_cross_validation

print('k-fold:',k_cross_validation)
print('The precision of cnn classification is :', np.round(cnn_pre_total,4))
print('The recall of cnn classification is :', np.round(cnn_rec_total,4))
print('The fscore of cnn classification is :', np.round(cnn_fsc_total,4))
print('The accuracy of cnn classification of k fold validation is:', np.round(cnn_cvscore,4))

record={'precision':np.round(cnn_pre_total, 4),
        'recall':np.round(cnn_rec_total, 4),
        'fscore':np.round(cnn_fsc_total, 4)
        }
rec=pd.DataFrame(record)
acc=pd.DataFrame(np.round(cnn_cvscore,4),columns=['acc'])
rec=pd.concat([rec,acc],axis=1)
rec.to_csv(r'C:\Users\wanqikang\Desktop\CNV实验\result.csv')

