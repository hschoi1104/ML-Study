# Chapter 5

## ch5.1
### 학습 모델 저장하고 재사용하기


```python
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np

data = np.loadtxt('./data.csv',delimiter=',',unpack=True, dtype = 'float32')

x_data = np.transpose(data[0:2])
y_data = np.transpose(data[2:])
```

    WARNING:tensorflow:From c:\python3.7\lib\site-packages\tensorflow_core\python\compat\v2_compat.py:65: disable_resource_variables (from tensorflow.python.ops.variable_scope) is deprecated and will be removed in a future version.
    Instructions for updating:
    non-resource variables are not supported in the long term
    

#### 신경망 모델 구성하기


```python
global_step = tf.Variable(0, trainable=False, name='global_step')
```


```python
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

W1 = tf.Variable(tf.random_uniform([2, 10], -1., 1.))
L1 = tf.nn.relu(tf.matmul(X, W1))

W2 = tf.Variable(tf.random_uniform([10, 20], -1., 1.))
L2 = tf.nn.relu(tf.matmul(L1, W2))

W3 = tf.Variable(tf.random_uniform([20, 3], -1., 1.))
model = tf.matmul(L2, W3)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y, logits=model))

optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
train_op = optimizer.minimize(cost, global_step=global_step)
```

#### 모델 저장


```python
sess = tf.Session()
saver = tf.train.Saver(tf.global_variables())
```

#### 기존에 학습한 모델이 있다면 가져오고 아니면 변수 초기화


```python
ckpt = tf.train.get_checkpoint_state('./model')
if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
    saver.restore(sess, ckpt.model_checkpoint_path)
else:
    sess.run(tf.global_variables_initializer())
```

    WARNING:tensorflow:From <ipython-input-5-748d720ec2ff>:2: checkpoint_exists (from tensorflow.python.training.checkpoint_management) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use standard file APIs to check for files with this prefix.
    INFO:tensorflow:Restoring parameters from ./model\dnn.ckpt-6
    

#### 학습 실행
#### 학습은 2번씩 만 실행하고 global_step 변수는 값이 저장됨


```python
for step in range(2):
    sess.run(train_op, feed_dict={X:x_data,Y: y_data})
    
    print('Step: %d, ' %sess.run(global_step), 'Cost: %.3f'%sess.run(cost, feed_dict={X: x_data, Y:y_data}))
```

    Step: 7,  Cost: 0.850
    Step: 8,  Cost: 0.821
    


```python
saver.save(sess, './model/dnn.ckpt', global_step=global_step)
```




    './model/dnn.ckpt-8'



#### 결과 확인


```python
prediction = tf.argmax(model, 1)
target = tf.argmax(Y,1)
print('예측값:',sess.run(prediction, feed_dict={X: x_data}))
print('실제값:',sess.run(target, feed_dict={Y: y_data}))
is_correct = tf.equal(prediction, target)
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
print('정확도: %.2f'%sess.run(accuracy * 100, feed_dict={X: x_data, Y: y_data}))
```

    예측값: [0 1 2 0 0 2]
    실제값: [0 1 2 0 0 2]
    정확도: 100.00
    

## ch5.2
### 텐서보드 사용하기


```python
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np

data = np.loadtxt('./data.csv',delimiter=',',unpack=True, dtype = 'float32')

x_data = np.transpose(data[0:2])
y_data = np.transpose(data[2:])
```

    WARNING:tensorflow:From c:\python3.7\lib\site-packages\tensorflow_core\python\compat\v2_compat.py:65: disable_resource_variables (from tensorflow.python.ops.variable_scope) is deprecated and will be removed in a future version.
    Instructions for updating:
    non-resource variables are not supported in the long term
    

#### with tf.name_scope('name') 로 묶어 텐서보드에서 한 계층 내부를 표현
#### name='W1' 텐서보드에서 해당 이름의 변수 확인 용이


```python
global_step = tf.Variable(0, trainable=False, name='global_step')

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

with tf.name_scope('layer1'):
    W1 = tf.Variable(tf.random_uniform([2,10],-1.,1.),name='W1')
    L1 = tf.nn.relu(tf.matmul(X, W1))
with tf.name_scope('layer2'):
    W2 = tf.Variable(tf.random_uniform([10,20],-1.,1.),name='W2')
    L2 = tf.nn.relu(tf.matmul(L1, W2))
with tf.name_scope('output'):
    W3 = tf.Variable(tf.random_uniform([20,3],-1.,1.),name='W3')
    model = tf.matmul(L2,W3)
with tf.name_scope('optimizer'):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y,logits=model))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
    train_op = optimizer.minimize(cost, global_step=global_step)
```

#### 손실값 추적


```python
tf.summary.scalar('cost',cost)
```




    <tf.Tensor 'cost:0' shape=() dtype=string>




```python
sess = tf.Session()
saver = tf.train.Saver(tf.global_variables())

ckpt = tf.train.get_checkpoint_state('./model')
if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
    saver.restore(sess, ckpt.model_checkpoint_path)
else:
    sess.run(tf.global_variables_initializer())
```


```python
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter('./logs',sess.graph)
```


```python
for step in range(100):
    sess.run(train_op, feed_dict={X: x_data, Y: y_data})
    
    print('Step: %d, '%sess.run(global_step),'Cost:%.3f' %sess.run(cost, feed_dict={X: x_data, Y: y_data}))
    
    summary = sess.run(merged, feed_dict={X: x_data, Y: y_data})
    writer.add_summary(summary, global_step=sess.run(global_step))
```

    Step: 1,  Cost:3.235
    Step: 2,  Cost:3.018
    Step: 3,  Cost:2.809
    Step: 4,  Cost:2.610
    Step: 5,  Cost:2.419
    Step: 6,  Cost:2.238
    Step: 7,  Cost:2.066
    Step: 8,  Cost:1.902
    Step: 9,  Cost:1.746
    Step: 10,  Cost:1.600
    Step: 11,  Cost:1.463
    Step: 12,  Cost:1.337
    Step: 13,  Cost:1.224
    Step: 14,  Cost:1.125
    Step: 15,  Cost:1.043
    Step: 16,  Cost:0.979
    Step: 17,  Cost:0.930
    Step: 18,  Cost:0.896
    Step: 19,  Cost:0.873
    Step: 20,  Cost:0.860
    Step: 21,  Cost:0.852
    Step: 22,  Cost:0.849
    Step: 23,  Cost:0.848
    Step: 24,  Cost:0.848
    Step: 25,  Cost:0.848
    Step: 26,  Cost:0.847
    Step: 27,  Cost:0.845
    Step: 28,  Cost:0.843
    Step: 29,  Cost:0.840
    Step: 30,  Cost:0.836
    Step: 31,  Cost:0.832
    Step: 32,  Cost:0.827
    Step: 33,  Cost:0.821
    Step: 34,  Cost:0.814
    Step: 35,  Cost:0.808
    Step: 36,  Cost:0.800
    Step: 37,  Cost:0.793
    Step: 38,  Cost:0.785
    Step: 39,  Cost:0.778
    Step: 40,  Cost:0.771
    Step: 41,  Cost:0.765
    Step: 42,  Cost:0.758
    Step: 43,  Cost:0.752
    Step: 44,  Cost:0.747
    Step: 45,  Cost:0.741
    Step: 46,  Cost:0.736
    Step: 47,  Cost:0.731
    Step: 48,  Cost:0.727
    Step: 49,  Cost:0.723
    Step: 50,  Cost:0.719
    Step: 51,  Cost:0.715
    Step: 52,  Cost:0.711
    Step: 53,  Cost:0.708
    Step: 54,  Cost:0.704
    Step: 55,  Cost:0.700
    Step: 56,  Cost:0.696
    Step: 57,  Cost:0.692
    Step: 58,  Cost:0.687
    Step: 59,  Cost:0.683
    Step: 60,  Cost:0.678
    Step: 61,  Cost:0.673
    Step: 62,  Cost:0.669
    Step: 63,  Cost:0.664
    Step: 64,  Cost:0.660
    Step: 65,  Cost:0.656
    Step: 66,  Cost:0.652
    Step: 67,  Cost:0.648
    Step: 68,  Cost:0.644
    Step: 69,  Cost:0.640
    Step: 70,  Cost:0.636
    Step: 71,  Cost:0.632
    Step: 72,  Cost:0.629
    Step: 73,  Cost:0.625
    Step: 74,  Cost:0.622
    Step: 75,  Cost:0.619
    Step: 76,  Cost:0.616
    Step: 77,  Cost:0.613
    Step: 78,  Cost:0.609
    Step: 79,  Cost:0.606
    Step: 80,  Cost:0.604
    Step: 81,  Cost:0.601
    Step: 82,  Cost:0.599
    Step: 83,  Cost:0.596
    Step: 84,  Cost:0.593
    Step: 85,  Cost:0.591
    Step: 86,  Cost:0.588
    Step: 87,  Cost:0.586
    Step: 88,  Cost:0.584
    Step: 89,  Cost:0.582
    Step: 90,  Cost:0.580
    Step: 91,  Cost:0.578
    Step: 92,  Cost:0.576
    Step: 93,  Cost:0.574
    Step: 94,  Cost:0.573
    Step: 95,  Cost:0.571
    Step: 96,  Cost:0.570
    Step: 97,  Cost:0.569
    Step: 98,  Cost:0.568
    Step: 99,  Cost:0.567
    Step: 100,  Cost:0.566
    


```python
saver.save(sess, './model/dnn.ckpt', global_step=global_step)

prediction = tf.argmax(model, 1)
target = tf.argmax(Y, 1)
print('예측값:', sess.run(prediction, feed_dict={X: x_data}))
print('실제값:', sess.run(target, feed_dict={Y: y_data}))

is_correct = tf.equal(prediction, target)
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
print('정확도: %.2f' % sess.run(accuracy * 100, feed_dict={X: x_data, Y: y_data}))
```

    예측값: [0 1 2 0 0 2]
    실제값: [0 1 2 0 0 2]
    정확도: 100.00
    

![image.png](attachment:image.png)

![image.png](attachment:image.png)


```python

```
