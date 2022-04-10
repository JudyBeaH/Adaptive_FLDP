from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from Global_Parameter import *
from EPS_round import *
import tensorflow as tf
import numpy as np
import csv
import time
import struct
import pickle
import copy


import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'



batch_size = 600

l2_norm_clip = 4

per_clip = l2_norm_clip/batch_size




from tensorflow_privacy.privacy.analysis import compute_dp_sgd_privacy
from tensorflow_privacy.privacy.optimizers.dp_optimizer import DPGradientDescentGaussianOptimizer


# model = tf.keras.Sequential([
#     tf.keras.layers.Conv2D(16, 3,
#                            strides=2,
#                            padding='same',
#                            activation='relu',
#                            input_shape=(28, 28, 1)),
#     tf.keras.layers.MaxPool2D(2, 2),
#     tf.keras.layers.Conv2D(32, 3,
#                            strides=2,
#                            padding='valid',
#                            activation='relu'),
#     tf.keras.layers.MaxPool2D(2, 2),
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dense(32, activation='relu'),
#     tf.keras.layers.Dense(10, activation='softmax')
# ])


# ######################## without bias
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu",use_bias=False),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu",use_bias=False),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(10, activation="softmax",use_bias=False),
])



# model = tf.keras.Sequential([
#     tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
#     tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
#     tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
#     tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dropout(0.5),
#     tf.keras.layers.Dense(10, activation="softmax"),
# ])

# model = tf.keras.Sequential([
#     tf.keras.layers.Dense(600, activation='relu',use_bias=False),
#     tf.keras.layers.Dense(100, activation='relu',use_bias=False),
#     tf.keras.layers.Dense(10, activation='softmax',use_bias=False)
# ])

#----------------------------------------------------------------------#
#read return说明：
#img: [60000,28,28,1],60000个样本，28x28x1
#lbl:[60000,],样本对应的label
#----------------------------------------------------------------------#
def read(dataset = "training", path = "."):

    if dataset == "training":
        fname_img = os.path.join(path, 'train-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels-idx1-ubyte')
    elif dataset == "testing":
        fname_img = os.path.join(path, 't10k-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 't10k-labels-idx1-ubyte')
    else:
        raise (ValueError, "dataset must be 'testing' or 'training'")

    print(fname_lbl)

    # Load everything in some numpy arrays
    with open(fname_lbl, 'rb') as flbl:
        magic, num = struct.unpack(">II", flbl.read(8))
        lbl = np.fromfile(flbl, dtype=np.int8)

    with open(fname_img, 'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        img = np.fromfile(fimg, dtype=np.uint8).reshape(len(lbl), rows, cols)


    # Reshape and normalize

    img = np.reshape(img, [img.shape[0], img.shape[1], img.shape[2],1])*1.0/255.0
    print("这里观察img的形状:",img.shape[0], img.shape[1], img.shape[2],dataset)
    print("这里观察img的形状:",lbl.shape[0],dataset)
    #img = np.reshape(img, [img.shape[0], img.shape[1]* img.shape[2]]) * 1.0 / 255.0

    return img, lbl



#----------------------------------------------------------------------#
#get_data() return说明：
#非降序x集（按label的序）,非降序y集，验证x集，验证y集，测试x集，测试y集
#----------------------------------------------------------------------#
def get_data():
    # load the data
    x_train, y_train = read('training', './MNIST_original')
    x_test, y_test = read('testing', './MNIST_original')

    # create validation set,size:10000
    x_vali = list(x_train[50000:].astype(float))
    y_vali = list(y_train[50000:].astype(float))

    # create train_set,size:50000
    x_train = x_train[:50000].astype(float)
    y_train = y_train[:50000].astype(float)

    # sort train set (to make federated learning non i.i.d.)
    #indices_train是label非降序的索引
    indices_train = np.argsort(y_train)
    sorted_x_train = list(x_train[indices_train]) #获得一个排序好的x_train
    sorted_y_train = list(y_train[indices_train]) #获得一个排序好的y_train
    
    # create a test set
    x_test = list(x_test.astype(float))
    y_test = list(y_test.astype(float))

    return sorted_x_train, sorted_y_train, x_vali, y_vali, x_test, y_test




client_batch=5
total_local_iter=100
client_set = pickle.load(open('./DATA/clients/' + str(total_client_num) + '_clients.pkl', 'rb'))
sorted_x_train, sorted_y_train, x_vali, y_vali, x_test, y_test = get_data()

data_set_asarray = np.asarray(sorted_x_train)
label_set_asarray = np.asarray(sorted_y_train)


accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
accuracy_test = tf.keras.metrics.SparseCategoricalAccuracy()

######### for private mode;
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False,reduction=tf.keras.losses.Reduction.NONE)
######## for benign model
#loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)

#optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
#optimizer = tf.keras.optimizers.Adam(learning_rate=0.05)
optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
#optimizer = tf.keras.optimizers.SGD(learning_rate=0.05,momentum=0.9)
#optimizer = tf.keras.optimizers.SGD(learning_rate=0.001)



# dp_optimizer = DPGradientDescentGaussianOptimizer(
#     l2_norm_clip=l2_norm_clip,
#     noise_multiplier=noise_multiplier,
#     num_microbatches=batch_size,
#     learning_rate=learning_rate)
# dp_loss = tf.keras.losses.CategoricalCrossentropy(
#     from_logits=True, reduction=tf.losses.Reduction.NONE)
#
###先对模型进行一个实例化，获取初始的weight值
model.build(np.asarray(x_test).shape)


new_global_model = copy.deepcopy(model)
starting_time = time.time()
eps_global = eps_global_init
for round in range(rounds):
    ##############################  Server端      ###########################################
    #---------------------------------------------------------------------------------#
    #数据划分方式：shuffle，每个用户固定拥有500个数据，每轮选取partiClientNum个用户进行计算#
    perm = np.random.permutation(total_client_num) #对所有的client进行一个shuffle
    
    s = perm[0:parti_client_num].tolist()
    print("round:",round,"选取的client：",s)
    participating_clients_data = [client_set[k] for k in s]
    #---------------------------------------------------------------------------------#
    #Step 1
    #划分该轮的epsilon，用到vali_set（10000个样本）与old_global_model，并利用得到的准确率进行一个计算
    #in:
    #out:eps_round在t轮时
    old_global_model = copy.deepcopy(new_global_model)
    epsRoundAccount = EPS_round(x_vali,y_vali) #传入固定vali集,实例化
    t = round+1
    eps_round = EPS_round.RoundlyAccount(old_global_model,eps_global,t) #计算得该轮的隐私预算
    eps_global -= eps_round
    #---------------------------------------------------------------------------------#
    #Step 2
    #利用服务器的DLA对该轮选定的client进行攻击，获得一个预算分配的方案
    #输入：eps_round
    #输出：对每个client的隐私预算list: eps_clients[parti_client_num]
    ##############################  Server端  END   ###########################################
    
    ##############################  Client端        ###########################################
    for k_t in range(parti_client_num):
        #data_ind = np.split(np.asarray(participating_clients_data[k_t]), client_batch, 0)
        data_ind = np.split(np.asarray(participating_clients_data[k_t]), 100, 0)
        #data_ind = np.split(np.asarray(participating_clients_data[k_t]), 50, 0)

        model =copy.deepcopy(old_global_model)
        #for local_iter in range(total_local_iter):

        for local_iter in range(len(data_ind)):

            batch_ind=data_ind[local_iter]

            x = data_set_asarray[[int(j) for j in batch_ind]]
            y = label_set_asarray[[int(j) for j in batch_ind]]



            ##################################################################################
            ## fed-cdp, instance level noise and differential privacy protection
            #for idx in range(int(x.shape[0]/50)):

            with tf.GradientTape(persistent=True) as tape:

                logits = model(x)
                loss_value = loss_fn(y, logits)



                #print (loss_value)
                for idx_sub in range(int(x.shape[0])):
                    cur_gradients = tape.gradient(loss_value[idx_sub], model.trainable_weights)
                    num_weights = len(cur_gradients)
                    if idx_sub==0:
                        per_gradient = [tf.expand_dims(cur_gradients[i], -1) for i in range(num_weights)]
                    else:
                        per_gradient = [tf.concat([per_gradient[i], tf.expand_dims(cur_gradients[i], -1)],axis=-1) for i in range(num_weights)]




                norms = [np.sqrt(np.sum(np.square(per_gradient[i]),axis=tuple(range(per_gradient[i].ndim)[:-1]),keepdims=True)) for i in range(num_weights)]




                factors = [norms[i] / l2_norm_clip for i in range(num_weights)]
                for i in range(num_weights):
                #for factor in factors:
                    factors[i][factors[i]<1.0]=1.0
                    # factor = tf.cond(factor < 1.0, lambda: 1.0,
                    #                  lambda: factor)  # get the clipping factor max(1,norm/clipping)


                #clipped_gradients = [per_gradient[i] / np.max([1,factors[i]]) for i in range(num_weights)]  # do clipping
                clipped_gradients = [per_gradient[i] / factors[i] for i in range(num_weights)]  # do clipping

                #if local_iter ==0:
                clipped_sum = [np.mean(clipped_gradients[i],-1) for i in range(num_weights)]
                    #logits_all = logits
                #else:
                #    clipped_sum = [clipped_sum[i] + np.mean(clipped_gradients[i],-1) for i in range(num_weights)]
                    #logits_all = tf.concat([logits_all,logits],axis=0)


            MeanClippedgradients = clipped_sum


            #
            # ##############if fixed clipping
            GaussianNoises = [
                1.0 / x.shape[0] * np.random.normal(loc=0.0, scale=float(noise_multiplier * l2_norm_clip),
                                                    size=MeanClippedgradients[i].shape) for i in
                range(num_weights)]  # layerwise gaussian noise



            Sanitized_gradients = [MeanClippedgradients[i] + GaussianNoises[i] for i in range(num_weights)]  # add gaussian noise


            optimizer.apply_gradients(zip(Sanitized_gradients, model.trainable_weights))


        if k_t==0:
            local_model =  [model.get_weights()[i] for i in range(num_weights)]
        else:
            local_model = [local_model[i] + model.get_weights()[i] for i in range(num_weights)]
    
    ##############################  Client端  END   ###########################################
    
    new_global_model.set_weights([local_model[i]/k_t for i in range(num_weights)]) #FedAvg

   







        #if round % 1 == 0 and local_iter %100 == 0:
        #:

            # ##########################################################
            # ######print mean clipped gradients
            # #for i,item in enumerate(gradients):
            # for i,item in enumerate(MeanClippedgradients):
            # #for i,item in enumerate(clipped_sum_nonoise):
            # #for i,item in enumerate(norms):
            # #for i,item in enumerate(per_gradient):
            # #     #print ('mean:',np.mean(tf.reshape(item,[1,-1])), 'max:',np.max(tf.reshape(item,[1,-1])),'l2:',tf.norm(tf.reshape(item,[1,-1])))
            #     l2norm = tf.norm(tf.reshape(item,[1,-1]))
            #     print('l2:',l2norm)
            #     max = tf.math.reduce_max(tf.reshape(item,[1,-1]))
            #     abs_max = tf.math.reduce_max(tf.math.abs(tf.reshape(item,[1,-1])))
            #     min = tf.math.reduce_min(tf.reshape(item, [1, -1]))
            #     mean = tf.math.reduce_mean(tf.reshape(item,[1,-1]))
            #     variance = tf.math.reduce_std(tf.reshape(item,[1,-1]))
            #     noise_C = np.mean(l2_norm_clip, axis=0)
            #
            #     # if i==4:
            #     #     print ('clipped:',tf.norm(tf.reshape(clipped_gradients[i],[1,-1])))
            #     #     print ('raw',tf.norm(tf.reshape(gradients[i],[1,-1])))
            #     #     # print('clipped:', clipped_gradients[i])
            #     #     # print('raw', gradients[i])
            #     with open('meanclipped_%s.csv'%i, mode='a') as norm_mean_file:
            #         writer_train = csv.writer(norm_mean_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            #
            #         writer_train.writerow([round,local_iter,l2norm,max,abs_max,min,mean,variance,noise_C])

            ##########################################################
            ######print noisy gradients
            #for i, item in enumerate(Sanitized_gradients):
            #for i,item in enumerate(gradients):
            #for i,item in enumerate(per_gradients):
            #    l2norm = tf.norm(tf.reshape(item, [1, -1]))
            #    print('l2:', l2norm)
            #    max = tf.math.reduce_max(tf.reshape(item, [1, -1]))
            #    abs_max = tf.math.reduce_max(tf.math.abs(tf.reshape(item, [1, -1])))
            #    min = tf.math.reduce_min(tf.reshape(item, [1, -1]))
            #    mean = tf.math.reduce_mean(tf.reshape(item, [1, -1]))
            #    variance = tf.math.reduce_std(tf.reshape(item, [1, -1]))

            #    with open('sanitized_%s.csv' % i, mode='a') as norm_sani_file:
            #        writer_train = csv.writer(norm_sani_file, delimiter=',', quotechar='"',
            #                                  quoting=csv.QUOTE_MINIMAL)

            #        writer_train.writerow([round,local_iter, l2norm, max, abs_max, min, mean, variance])



    cur_time = time.time()
    duration = cur_time-starting_time
    if local_iter % 1 == 0:
        # for Sanitized_gradient in Sanitized_gradients:
        #    print(anitized_gradient)
        # for item in norms:
        #   print(item)
        # print (l2_norm_clip)

        # Update the state of the `accuracy` metric.
        logits_all = new_global_model(np.asarray(sorted_x_train)[:10000], 0)
        accuracy.update_state(np.asarray(sorted_y_train)[:10000],logits_all)


        #accuracy.update_state(y, logits_all)
        print("round:", round, "local iter:", local_iter)
        print("Training accuracy: %.5f" % accuracy.result())
        with open('training.csv', mode='a') as train_file:
            writer_train = csv.writer(train_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

            writer_train.writerow([round,local_iter,accuracy.result(),duration])

    if local_iter % 1 == 0:
        logits_test = new_global_model(np.asarray(x_test))
        accuracy_test.update_state(np.asarray(y_test), logits_test)
        #logits_test = new_global_model(np.asarray(x_vali))
        #accuracy_test.update_state(np.asarray(y_vali), logits_test)

        print("round:", round, "local iter:", local_iter)
        print("Test accuracy: %.5f" % accuracy_test.result())

        with open('test.csv', mode='a') as test_file:
            writer_test = csv.writer(test_file, delimiter=',')

            writer_test.writerow([round,local_iter,accuracy_test.result()])

    with open('sigma.csv', mode='a') as sigma_file:
        writer_sigma = csv.writer(sigma_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        # writer_sigma.writerow([epoch, step, 24/sqrt(sum(n*n for n in l2_norm_clip)/len(l2_norm_clip))])  #if per exampl
        writer_sigma.writerow([round, local_iter, 24.0 / l2_norm_clip])  # if batch

            # Reset the metric's state at the end of an global iter
    accuracy.reset_states()
    accuracy_test.reset_states()



















