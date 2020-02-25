# Chapter 10 - RNN

## ch10.2

### 단어 자동완성
- 영문자 4개로 된 단어를 학습시켜, 3글자만 주어지면 나머지 한 글자를 추천하여 단어를 완성하는 프로그램


```python
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np

char_arr = ['a', 'b', 'c', 'd', 'e', 'f', 'g',
            'h', 'i', 'j', 'k', 'l', 'm', 'n',
            'o', 'p', 'q', 'r', 's', 't', 'u',
            'v', 'w', 'x', 'y', 'z']

num_dic = {n: i for i, n in enumerate(char_arr)}
dic_len = len(num_dic)

# 학습에 사용할 단어
seq_data = ['word', 'wood', 'deep', 'dive', 'cold', 'cool', 'load', 'love', 'kiss', 'kind']
```

    WARNING:tensorflow:From c:\python3.7\lib\site-packages\tensorflow_core\python\compat\v2_compat.py:65: disable_resource_variables (from tensorflow.python.ops.variable_scope) is deprecated and will be removed in a future version.
    Instructions for updating:
    non-resource variables are not supported in the long term
    

#### 학습에 사용할 수 있는 형식으로 변호나해주는 유틸리티 함수 작성
1. 입력값용으로, 단어의 처음 세 글자의 알파벳 인덱스를 구한 배열을 만든다.
> input = [num_dic[n] for n in seq[:-1]]
2. 출력값용으로, 마지막 글자의 알파벳 인덱스를 구한다.
> target = num_dic[seq[-1]]
3. 입력값을 원-핫 인코딩으로 변환
> input_batch.append(np.eye(dic_len)[input])

#### 예시
`deep` 경우
- 입력으로 `d,e,e`를 취함.
- 알파벳의 인덱스를 구해 배열로 만들면 `[3,4,4]`가 됨
- 원핫 인코딩 하면

```
[[0. 0. 0. 1. 0. 0. 0....]
 [0. 0. 0. 1. 0. 0. 0....]
 [0. 0. 0. 0. 1. 0. 0....]]
 ```
- 실측값은 p의 인덱스인 15가 된다.

- 실측값은 원-핫 인코딩하지 않고 15를 그대로 사용

- `sparse_softmax_cross_entropy_with_logits`를 사용 할 예정이기 때문

- `sparse_softmax_cross_entropy_with_logits`:실측값,에 원-핫 인코딩을 사용하지 않아도 자동으로 변환


```python
def make_batch(seq_data):
    input_batch = []
    target_batch = []

    for seq in seq_data:
        input = [num_dic[n] for n in seq[:-1]]
        target = num_dic[seq[-1]]
        input_batch.append(np.eye(dic_len)[input])
        target_batch.append(target)

    return input_batch, target_batch
```

### 신경망 모델 구성
#### 하이퍼 파라미터 설정
- `sparse_softmax_cross_entropy_with_logits` 함수를 사용할 때 실측값인 Labels 값은 인덱스의 숫자를 그대로 사용하고, 예측 모델의 출력값은 인덱스의 원-핫 인코딩을 사용
- 예측모델의 출력값은 원-핫 인코딩을 사용


```python
learning_rate = 0.01
n_hidden = 128
total_epoch = 30
n_step = 3 # 3글자를 학습 할 것
n_input = n_class = dic_len # 원핫 인코딩을 이용하므로 `dic_len`과 일치
```

#### 모델 구성


```python
X = tf.placeholder(tf.float32, [None, n_step, n_input])
Y = tf.placeholder(tf.int32, [None]) 
# batch_size의 하나의 차원만 있음
# 원-핫 인코딩이 아니라 인덱스 숫자를 그대로 사용하기 때문 

W = tf.Variable(tf.random_normal([n_hidden, n_class]))
b = tf.Variable(tf.random_normal([n_class]))
```

#### RNN 셀 생성
- 여러 셀을 조합해 심층 신경망을 만들기 위해 두개의 RNN 셀 생성
- `DropoutWrapper`함수를 이용해 RNN에서도 과적합 방지를 위한 드롭아웃 기법 적용 가능


```python
cell1 = tf.nn.rnn_cell.BasicLSTMCell(n_hidden)
cell1 = tf.nn.rnn_cell.DropoutWrapper(cell1, output_keep_prob=0.5)
cell2 = tf.nn.rnn_cell.BasicLSTMCell(n_hidden)
```

    WARNING:tensorflow:From <ipython-input-5-c6b59c0bdf95>:1: BasicLSTMCell.__init__ (from tensorflow.python.ops.rnn_cell_impl) is deprecated and will be removed in a future version.
    Instructions for updating:
    This class is equivalent as tf.keras.layers.LSTMCell, and will be replaced by that in Tensorflow 2.0.
    

#### 심층순환신경망 DeepRNN
- `Multi_cell` 함수를 이용해 조합하고 `dynamic_rnn`을 이용해 심층 순환 신경망 만듬


```python
multi_cell = tf.nn.rnn_cell.MultiRNNCell([cell1, cell2])
# time_major=True
outputs, states = tf.nn.dynamic_rnn(multi_cell, X, dtype=tf.float32)
```

    WARNING:tensorflow:From <ipython-input-6-bdaa49857441>:1: MultiRNNCell.__init__ (from tensorflow.python.ops.rnn_cell_impl) is deprecated and will be removed in a future version.
    Instructions for updating:
    This class is equivalent as tf.keras.layers.StackedRNNCells, and will be replaced by that in Tensorflow 2.0.
    WARNING:tensorflow:From <ipython-input-6-bdaa49857441>:3: dynamic_rnn (from tensorflow.python.ops.rnn) is deprecated and will be removed in a future version.
    Instructions for updating:
    Please use `keras.layers.RNN(cell)`, which is equivalent to this API
    WARNING:tensorflow:From c:\python3.7\lib\site-packages\tensorflow_core\python\ops\rnn_cell_impl.py:735: Layer.add_variable (from tensorflow.python.keras.engine.base_layer) is deprecated and will be removed in a future version.
    Instructions for updating:
    Please use `layer.add_weight` method instead.
    WARNING:tensorflow:From c:\python3.7\lib\site-packages\tensorflow_core\python\ops\rnn_cell_impl.py:739: calling Zeros.__init__ (from tensorflow.python.ops.init_ops) with dtype is deprecated and will be removed in a future version.
    Instructions for updating:
    Call initializer instance with the dtype argument instead of passing it to the constructor
    

#### 최종 출력층


```python
outputs = tf.transpose(outputs, [1, 0, 2])
outputs = outputs[-1]
model = tf.matmul(outputs, W) + b
```

#### 손실함수와 최적화 함수 설정 및 모델 구성 마무리


```python
cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=model, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
```

### 신경망 학습
#### 학습 진행



```python
sess = tf.Session()
sess.run(tf.global_variables_initializer())

input_batch, target_batch = make_batch(seq_data)

for epoch in range(total_epoch):
    _, loss = sess.run([optimizer, cost],feed_dict={X: input_batch, Y: target_batch})

    print('Epoch:', '%04d' % (epoch + 1),'cost =', '{:.6f}'.format(loss))

print('최적화 완료!')
```

    Epoch: 0001 cost = 2.562865
    Epoch: 0002 cost = 1.905663
    Epoch: 0003 cost = 1.377334
    Epoch: 0004 cost = 0.786262
    Epoch: 0005 cost = 0.801324
    Epoch: 0006 cost = 0.485126
    Epoch: 0007 cost = 0.557528
    Epoch: 0008 cost = 0.461366
    Epoch: 0009 cost = 0.297106
    Epoch: 0010 cost = 0.512536
    Epoch: 0011 cost = 0.412064
    Epoch: 0012 cost = 0.277375
    Epoch: 0013 cost = 0.329555
    Epoch: 0014 cost = 0.296244
    Epoch: 0015 cost = 0.344352
    Epoch: 0016 cost = 0.290036
    Epoch: 0017 cost = 0.206698
    Epoch: 0018 cost = 0.080102
    Epoch: 0019 cost = 0.034261
    Epoch: 0020 cost = 0.129824
    Epoch: 0021 cost = 0.055783
    Epoch: 0022 cost = 0.074521
    Epoch: 0023 cost = 0.031576
    Epoch: 0024 cost = 0.042749
    Epoch: 0025 cost = 0.012007
    Epoch: 0026 cost = 0.028949
    Epoch: 0027 cost = 0.062839
    Epoch: 0028 cost = 0.005002
    Epoch: 0029 cost = 0.008009
    Epoch: 0030 cost = 0.003160
    최적화 완료!
    

##### 결과 확인


```python
# 레이블값이 정수이므로 예측값도 정수로 변경해줍니다.
prediction = tf.cast(tf.argmax(model, 1), tf.int32)
# one-hot 인코딩이 아니므로 입력값을 그대로 비교합니다.
prediction_check = tf.equal(prediction, Y)
accuracy = tf.reduce_mean(tf.cast(prediction_check, tf.float32))

input_batch, target_batch = make_batch(seq_data)

predict, accuracy_val = sess.run([prediction, accuracy],
                                 feed_dict={X: input_batch, Y: target_batch})

predict_words = []
for idx, val in enumerate(seq_data):
    last_char = char_arr[predict[idx]]
    predict_words.append(val[:3] + last_char)

print('\n=== 예측 결과 ===')
print('입력값:', [w[:3] + ' ' for w in seq_data])
print('예측값:', predict_words)
print('정확도:', accuracy_val)
```

    
    === 예측 결과 ===
    입력값: ['wor ', 'woo ', 'dee ', 'div ', 'col ', 'coo ', 'loa ', 'lov ', 'kis ', 'kin ']
    예측값: ['word', 'wood', 'deep', 'dive', 'cold', 'cool', 'load', 'love', 'kiss', 'kind']
    정확도: 1.0
    
