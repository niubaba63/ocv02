import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
 
#导入数据集
mnist = input_data.read_data_sets("MNIST_data",one_hot=True)#如果本地没有数据集，此语句会自动下载到对应的文件夹位置，不过网速较慢，不建议
print("Training data size:", mnist.train.num_examples)
#每个批次的大小
batch_size = 100
#计算一共需要多少个批次
n_batch = mnist.train.num_examples // batch_size
#创建两个placeholder
x = tf.placeholder(tf.float32,[None,784])
y = tf.placeholder(tf.float32,[None,10])
#创建一个简单的神经网络
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
prediction = tf.nn.softmax(tf.matmul(x,W) + b)
#二次代价函数
loss = tf.reduce_mean(tf.square(y-prediction))
#交叉熵损失函数，可以与二次代价函数对比一下那个效果好，运行时只能保留一个
#loss = -tf.reduce_sum(y_*tf.log(y))
 
#使用梯度下降法训练
train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)
#结果存放在一个布尔类型列表中
correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(prediction,1))#对比预测结果的标签是否一致，一致为True，不同为False
#预测准确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))#将布尔型转化为0.0-1.0之间的数值,True为1.0，False为0.0
#变量初始化
init = tf.global_variables_initializer()

with tf.Session(config=tf.ConfigProto(device_count={'gpu':0})) as sess:
#with tf.Session() as sess:
    sess.run(init)
    for epoch in range(21):
        for batch in range(n_batch):
            batch_x,batch_y = mnist.train.next_batch(batch_size)#
            sess.run(train_step,feed_dict={x:batch_x,y:batch_y})
        acc = sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels})
        print('Iter' + str(epoch) + ',Test Accuaracy' + str(acc))
