# Chapter 10 - RNN

## ch10.1

### MINIST 를 RNN으로


```python
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./mnist/data/", one_hot=True)
```

    WARNING:tensorflow:From c:\python3.7\lib\site-packages\tensorflow_core\python\compat\v2_compat.py:65: disable_resource_variables (from tensorflow.python.ops.variable_scope) is deprecated and will be removed in a future version.
    Instructions for updating:
    non-resource variables are not supported in the long term
    WARNING:tensorflow:From <ipython-input-1-a988c6987768>:5: read_data_sets (from tensorflow.examples.tutorials.mnist.input_data) is deprecated and will be removed in a future version.
    Instructions for updating:
    Please use alternatives such as: tensorflow_datasets.load('mnist')
    WARNING:tensorflow:From c:\python3.7\lib\site-packages\tensorflow_core\examples\tutorials\mnist\input_data.py:297: _maybe_download (from tensorflow.examples.tutorials.mnist.input_data) is deprecated and will be removed in a future version.
    Instructions for updating:
    Please write your own downloading logic.
    WARNING:tensorflow:From c:\python3.7\lib\site-packages\tensorflow_core\examples\tutorials\mnist\input_data.py:299: _extract_images (from tensorflow.examples.tutorials.mnist.input_data) is deprecated and will be removed in a future version.
    Instructions for updating:
    Please use tf.data to implement this functionality.
    Extracting ./mnist/data/train-images-idx3-ubyte.gz
    WARNING:tensorflow:From c:\python3.7\lib\site-packages\tensorflow_core\examples\tutorials\mnist\input_data.py:304: _extract_labels (from tensorflow.examples.tutorials.mnist.input_data) is deprecated and will be removed in a future version.
    Instructions for updating:
    Please use tf.data to implement this functionality.
    Extracting ./mnist/data/train-labels-idx1-ubyte.gz
    WARNING:tensorflow:From c:\python3.7\lib\site-packages\tensorflow_core\examples\tutorials\mnist\input_data.py:112: _dense_to_one_hot (from tensorflow.examples.tutorials.mnist.input_data) is deprecated and will be removed in a future version.
    Instructions for updating:
    Please use tf.one_hot on tensors.
    Extracting ./mnist/data/t10k-images-idx3-ubyte.gz
    Extracting ./mnist/data/t10k-labels-idx1-ubyte.gz
    WARNING:tensorflow:From c:\python3.7\lib\site-packages\tensorflow_core\examples\tutorials\mnist\input_data.py:328: _DataSet.__init__ (from tensorflow.examples.tutorials.mnist.input_data) is deprecated and will be removed in a future version.
    Instructions for updating:
    Please use alternatives such as official/mnist/_DataSet.py from tensorflow/models.
    

#### 하이퍼 파라미터, 변수, 출력층 가중치와 편향

- RNN은 순서가 있는 데이터를 다루므로 한 번에 입력 받을 개수와 총 몇단계로 이뤄진 데이터를 받을지 설정 해야함. 이를위해 `n_input`, `n_step` 변수 사용

- 출력 값은 MINIST의 분류인 0~9까지 10개의 숫자를 원-핫 인코딩으로 표현


```python
learning_rate = 0.001
total_epoch = 30
batch_size = 128

n_input = 28 # 가로픽셀 수
n_step = 28 # 세로필셀 수, 입력단계
n_hidden = 128
n_class = 10

X = tf.placeholder(tf.float32, [None, n_step, n_input])
Y = tf.placeholder(tf.float32, [None, n_class])

W = tf.Variable(tf.random_normal([n_hidden, n_class]))
b = tf.Variable(tf.random_normal([n_class]))
```

#### `n_hidden`개의 출력 값을 가지는 RNN 셀 생성
- `BasicRNNCell` 말고도 `BasicLSTMCell`,`GRUCell`등 다양한 방식의 셀이 있다.


```python
cell = tf.nn.rnn_cell.BasicRNNCell(n_hidden)
```

    WARNING:tensorflow:From <ipython-input-3-e006f918b220>:1: BasicRNNCell.__init__ (from tensorflow.python.ops.rnn_cell_impl) is deprecated and will be removed in a future version.
    Instructions for updating:
    This class is equivalent as tf.keras.layers.SimpleRNNCell, and will be replaced by that in Tensorflow 2.0.
    

#### `dynamic_rnn`함수를 이용해 RNN 신경망 완성
- RNN 셀과 입력값 자료형을 넣어주면 신경망 생성 가능


```python
outputs, states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
```

    WARNING:tensorflow:From <ipython-input-4-f7b88a02a855>:1: dynamic_rnn (from tensorflow.python.ops.rnn) is deprecated and will be removed in a future version.
    Instructions for updating:
    Please use `keras.layers.RNN(cell)`, which is equivalent to this API
    WARNING:tensorflow:From c:\python3.7\lib\site-packages\tensorflow_core\python\ops\rnn_cell_impl.py:456: Layer.add_variable (from tensorflow.python.keras.engine.base_layer) is deprecated and will be removed in a future version.
    Instructions for updating:
    Please use `layer.add_weight` method instead.
    WARNING:tensorflow:From c:\python3.7\lib\site-packages\tensorflow_core\python\ops\rnn_cell_impl.py:460: calling Zeros.__init__ (from tensorflow.python.ops.init_ops) with dtype is deprecated and will be removed in a future version.
    Instructions for updating:
    Call initializer instance with the dtype argument instead of passing it to the constructor
    

#### 최종 출력값
- 결과를 원-핫 인코딩 형태로 만들것이므로 손실 함수로 `tf.nn.softmax_cross_entropy_with_logits_v2` 사용
- 이 함수를 사용하기 위해 최종 결괏값이 실측값 Y와 동일한 형태인 `[batch_size,n_class]` 이여야 한다.
- 하지만 RNN 신경망에서 출력값은 `[batch_size,n_step,n_hidden]` 형태이다.??


```python
# outputs : [batch_size, n_step, n_hidden]
#        -> [n_step, batch_size, n_hidden]
outputs = tf.transpose(outputs, [1, 0, 2])
#        -> [batch_size, n_hidden]
outputs = outputs[-1]
```

#### `y = X*W+b` 를 이용해서 최종 결괏값을 만듬


```python
model = tf.matmul(outputs, W) + b
```

#### 손실값 구하고 신경망 최적화


```python
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=model, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
```

#### 학습시작
- X 데이터를 RNN 입력 데이터에 맞게 `[batch_size, n_step, n_input]` 형태로 변환한다
        

> batch_xs = batch_xs.reshape((batch_size, n_step, n_input))



```python
sess = tf.Session()
sess.run(tf.global_variables_initializer())

total_batch = int(mnist.train.num_examples/batch_size)

for epoch in range(total_epoch):
    total_cost = 0

    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        batch_xs = batch_xs.reshape((batch_size, n_step, n_input))

        _, cost_val = sess.run([optimizer, cost],
                               feed_dict={X: batch_xs, Y: batch_ys})
        total_cost += cost_val

    print('Epoch:', '%04d' % (epoch + 1),
          'Avg. cost =', '{:.3f}'.format(total_cost / total_batch))

print('최적화 완료!')
```

    Epoch: 0001 Avg. cost = 0.516
    Epoch: 0002 Avg. cost = 0.231
    Epoch: 0003 Avg. cost = 0.184
    Epoch: 0004 Avg. cost = 0.160
    Epoch: 0005 Avg. cost = 0.136
    Epoch: 0006 Avg. cost = 0.131
    Epoch: 0007 Avg. cost = 0.120
    Epoch: 0008 Avg. cost = 0.112
    Epoch: 0009 Avg. cost = 0.110
    Epoch: 0010 Avg. cost = 0.104
    Epoch: 0011 Avg. cost = 0.100
    Epoch: 0012 Avg. cost = 0.092
    Epoch: 0013 Avg. cost = 0.095
    Epoch: 0014 Avg. cost = 0.090
    Epoch: 0015 Avg. cost = 0.084
    Epoch: 0016 Avg. cost = 0.077
    Epoch: 0017 Avg. cost = 0.085
    Epoch: 0018 Avg. cost = 0.075
    Epoch: 0019 Avg. cost = 0.079
    Epoch: 0020 Avg. cost = 0.080
    Epoch: 0021 Avg. cost = 0.076
    Epoch: 0022 Avg. cost = 0.063
    Epoch: 0023 Avg. cost = 0.073
    Epoch: 0024 Avg. cost = 0.064
    Epoch: 0025 Avg. cost = 0.071
    Epoch: 0026 Avg. cost = 0.070
    Epoch: 0027 Avg. cost = 0.065
    Epoch: 0028 Avg. cost = 0.064
    Epoch: 0029 Avg. cost = 0.062
    Epoch: 0030 Avg. cost = 0.067
    최적화 완료!
    

#### 결과 확인


```python
is_correct = tf.equal(tf.argmax(model, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

test_batch_size = len(mnist.test.images)
test_xs = mnist.test.images.reshape(test_batch_size, n_step, n_input)
test_ys = mnist.test.labels

print('정확도:', sess.run(accuracy,
                       feed_dict={X: test_xs, Y: test_ys}))
```

    정확도: 0.9719
    


```python

```
