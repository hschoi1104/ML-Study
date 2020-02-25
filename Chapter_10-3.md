# Chapter 10 - RNN

## ch10.3

### Sequence to Sequence

- 입력을 위한 신경망인 인코더와 출력을 위한 신경망인 디코더로 구성

- 인코더는 원문을 디코더는 인코더의 결과물을 입력

- 그 후 디코더의 결과물을 변역된 결과물과 비교하면서 학습

- 네글자의 영어 단어를 입력받아 두글자의 한글단어로 번역하는 프로그램

#### 심볼

- Sequence to Sequence 모델에는 몇가지 특수한 심볼이 필요함.

1. 디코더에 입력이 시작됨을 알려주는 심볼

2. 디코더의 출력이 끝났음을 알려주는 심볼

3. 빈데이터를 채울 때 사용하는 의미없는 심볼

- 여기서 심볼은 `s`,`e`,`p`로 처리

#### 데이터 생성

- 원-핫 인코딩 형식으로 바꿔야 하므로 영어 알파벳과 한글들을 나열한 뒤 한글자씩 배열에 집어넣음
- 배열에 넣은 글자들을 연관 배열(키/값) 형태로 변경한다.


```python
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np

# 원-핫 인코딩 형식으로 바꿔야 하므로 영어 알파벳과 한글들을 나열한 뒤 한글자씩 배열에 집어넣음
char_arr = [c for c in 'SEPabcdefghijklmnopqrstuvwxyz단어나무놀이소녀키스사랑']
# 글자:인덱스 형태의 연관 배열로 변경
num_dic = {n: i for i, n in enumerate(char_arr)}
dic_len = len(num_dic)
seq_data = [['word', '단어'], ['wood', '나무'],
            ['game', '놀이'], ['girl', '소녀'],
            ['kiss', '키스'], ['love', '사랑']]
```

    WARNING:tensorflow:From c:\python3.7\lib\site-packages\tensorflow_core\python\compat\v2_compat.py:65: disable_resource_variables (from tensorflow.python.ops.variable_scope) is deprecated and will be removed in a future version.
    Instructions for updating:
    non-resource variables are not supported in the long term
    

#### 데이터 확인


```python
print("====char_arr====")
print(char_arr)
print("====num_dic====")
print(num_dic)
print("====dic_len====")
print(dic_len)
```

    ====char_arr====
    ['S', 'E', 'P', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '단', '어', '나', '무', '놀', '이', '소', '녀', '키', '스', '사', '랑']
    ====num_dic====
    {'S': 0, 'E': 1, 'P': 2, 'a': 3, 'b': 4, 'c': 5, 'd': 6, 'e': 7, 'f': 8, 'g': 9, 'h': 10, 'i': 11, 'j': 12, 'k': 13, 'l': 14, 'm': 15, 'n': 16, 'o': 17, 'p': 18, 'q': 19, 'r': 20, 's': 21, 't': 22, 'u': 23, 'v': 24, 'w': 25, 'x': 26, 'y': 27, 'z': 28, '단': 29, '어': 30, '나': 31, '무': 32, '놀': 33, '이': 34, '소': 35, '녀': 36, '키': 37, '스': 38, '사': 39, '랑': 40}
    ====dic_len====
    41
    

#### 원-핫 인코딩 형식으로 만들어주는 유틸리티 함수

- `인코더입력값`, `디코더 입력값`, `디코더 출력값` 총 세 개로 구성


```python
def make_batch(seq_data):
    input_batch = []
    output_batch = []
    target_batch = []

    for seq in seq_data:
        #['word', '단어']
        #print("seq==========================")
        #print(seq)
        # 인코더 셀의 입력값을 위해 입력 단어를 한굴자씩 떼어 배열로 만듬
        # input:[25, 17, 20, 6]
        input = [num_dic[n] for n in seq[0]]
        print("input:")
        print(input)
        # 디코더 셀의 입력값을 위해 출력 단어의 글자들을 배열로 만들고, 시작을 나타내는 심볼 'S'를 맨 앞에 붙임
        # output: [0, 29, 30]
        output = [num_dic[n] for n in ('S' + seq[1])]
        print("output:")
        print(output)
        # target: [29, 30, 1]
        # 디코더 셀의 출력값을 만들고, 출력의 끝을 알려주는 심볼 'E'를 마지막에 붙임
        target = [num_dic[n] for n in (seq[1] + 'E')]
        print("target: ")
        print(target)
        # 원-핫 인코딩
        # 4*41 2차원배열
        input_batch.append(np.eye(dic_len)[input])
        #print("input_batch: ")
        #print(input_batch)
        # 3*41 2차원배열
        output_batch.append(np.eye(dic_len)[output])
        #print("output_batch: ")
        #print(output_batch)
        
        # [[29,30,1]] 의 형태
        # 인덱스 숫자를 사용(손실 함수 때문에)
        target_batch.append(target)
        print("target_batch: ")
        print(target_batch)

    return input_batch, output_batch, target_batch

```

#### 사용할 하이퍼파라미터,플레이스홀더,입출력변수용 수치 정의
- time steps : 같은 배치 때 입력되는 데이터 글자 수 , 단계


```python
learning_rate = 0.01
n_hidden = 128 # RNN 은닉층의 갯수
total_epoch = 100
n_class = n_input = dic_len

# 인코더와 디코더의 입력값 형식
# [batch size, time steps, input size]
enc_input = tf.placeholder(tf.float32, [None, None, n_input])
dec_input = tf.placeholder(tf.float32, [None, None, n_input])
# 디코더의 출력값 형식
# [batch size, time steps]
targets = tf.placeholder(tf.int64, [None, None])
```

#### 신경망 모델 구성
- 기본 셀 사용
- 각 셀에 드롭아웃 적용

#### 인코더 셀


```python
with tf.variable_scope('encode'):
    enc_cell = tf.nn.rnn_cell.BasicRNNCell(n_hidden)
    enc_cell = tf.nn.rnn_cell.DropoutWrapper(enc_cell, output_keep_prob=0.5)

    outputs, enc_states = tf.nn.dynamic_rnn(enc_cell, enc_input, dtype=tf.float32)
```

    WARNING:tensorflow:From <ipython-input-5-52b1601d878a>:2: BasicRNNCell.__init__ (from tensorflow.python.ops.rnn_cell_impl) is deprecated and will be removed in a future version.
    Instructions for updating:
    This class is equivalent as tf.keras.layers.SimpleRNNCell, and will be replaced by that in Tensorflow 2.0.
    WARNING:tensorflow:From <ipython-input-5-52b1601d878a>:5: dynamic_rnn (from tensorflow.python.ops.rnn) is deprecated and will be removed in a future version.
    Instructions for updating:
    Please use `keras.layers.RNN(cell)`, which is equivalent to this API
    WARNING:tensorflow:From c:\python3.7\lib\site-packages\tensorflow_core\python\ops\rnn_cell_impl.py:456: Layer.add_variable (from tensorflow.python.keras.engine.base_layer) is deprecated and will be removed in a future version.
    Instructions for updating:
    Please use `layer.add_weight` method instead.
    WARNING:tensorflow:From c:\python3.7\lib\site-packages\tensorflow_core\python\ops\rnn_cell_impl.py:460: calling Zeros.__init__ (from tensorflow.python.ops.init_ops) with dtype is deprecated and will be removed in a future version.
    Instructions for updating:
    Call initializer instance with the dtype argument instead of passing it to the constructor
    

#### 디코더 셀
- 디코더를 만들 때 초기 값으로 인코더의 최종 상태 값을 넣어줘야 한다.
- **인코더에서 계산한 상태를 디코더로 전파하는 것이기 때문**
- 텐서플로 `dynamic_rnn`에 `initial_state=enc_states`옵션으로 간단하게 처리


```python
with tf.variable_scope('decode'):
    dec_cell = tf.nn.rnn_cell.BasicRNNCell(n_hidden)
    dec_cell = tf.nn.rnn_cell.DropoutWrapper(dec_cell, output_keep_prob=0.5)

    # Seq2Seq 모델은 인코더 셀의 최종 상태값을
    # 디코더 셀의 초기 상태값으로 넣어주는 것이 핵심.
    outputs, dec_states = tf.nn.dynamic_rnn(dec_cell, dec_input, initial_state=enc_states, dtype=tf.float32)
```

#### 출력층만들고 손실 함수와 최적화 함수를 구성

#### layers.dence
- 완전 연결계층을 만들어준다.


```python
model = tf.layers.dense(outputs, n_class, activation=None)

cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=model, labels=targets))

optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
```

    WARNING:tensorflow:From <ipython-input-7-92f931a32b6b>:1: dense (from tensorflow.python.layers.core) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use keras.layers.Dense instead.
    WARNING:tensorflow:From c:\python3.7\lib\site-packages\tensorflow_core\python\layers\core.py:187: Layer.apply (from tensorflow.python.keras.engine.base_layer) is deprecated and will be removed in a future version.
    Instructions for updating:
    Please use `layer.__call__` method instead.
    

#### 학습

- `feed_dict`으로 전달하는 학습데이터에 인코더의 입력값, 디코더의 입력값과 출력값 이렇게 새 개를 넣음


```python
sess = tf.Session()
sess.run(tf.global_variables_initializer())

input_batch, output_batch, target_batch = make_batch(seq_data)

print(input_batch)
print(output_batch)
print(target_batch)
for epoch in range(total_epoch):
    _, loss = sess.run([optimizer, cost],
                       feed_dict={enc_input: input_batch,
                                  dec_input: output_batch,
                                  targets: target_batch})

    print('Epoch:', '%04d' % (epoch + 1),
          'cost =', '{:.6f}'.format(loss))

print('최적화 완료!')

```

    input:
    [25, 17, 20, 6]
    output:
    [0, 29, 30]
    target: 
    [29, 30, 1]
    target_batch: 
    [[29, 30, 1]]
    input:
    [25, 17, 17, 6]
    output:
    [0, 31, 32]
    target: 
    [31, 32, 1]
    target_batch: 
    [[29, 30, 1], [31, 32, 1]]
    input:
    [9, 3, 15, 7]
    output:
    [0, 33, 34]
    target: 
    [33, 34, 1]
    target_batch: 
    [[29, 30, 1], [31, 32, 1], [33, 34, 1]]
    input:
    [9, 11, 20, 14]
    output:
    [0, 35, 36]
    target: 
    [35, 36, 1]
    target_batch: 
    [[29, 30, 1], [31, 32, 1], [33, 34, 1], [35, 36, 1]]
    input:
    [13, 11, 21, 21]
    output:
    [0, 37, 38]
    target: 
    [37, 38, 1]
    target_batch: 
    [[29, 30, 1], [31, 32, 1], [33, 34, 1], [35, 36, 1], [37, 38, 1]]
    input:
    [14, 17, 24, 7]
    output:
    [0, 39, 40]
    target: 
    [39, 40, 1]
    target_batch: 
    [[29, 30, 1], [31, 32, 1], [33, 34, 1], [35, 36, 1], [37, 38, 1], [39, 40, 1]]
    [array([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0.]]), array([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0.]]), array([[0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0.]]), array([[0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0.]]), array([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0.]]), array([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0.]])]
    [array([[1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0.]]), array([[1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.,
            0., 0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            1., 0., 0., 0., 0., 0., 0., 0., 0.]]), array([[1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 1., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 1., 0., 0., 0., 0., 0., 0.]]), array([[1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 1., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 1., 0., 0., 0., 0.]]), array([[1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 1., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 1., 0., 0.]]), array([[1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 1., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 1.]])]
    [[29, 30, 1], [31, 32, 1], [33, 34, 1], [35, 36, 1], [37, 38, 1], [39, 40, 1]]
    Epoch: 0001 cost = 3.725879
    Epoch: 0002 cost = 2.628302
    Epoch: 0003 cost = 1.629524
    Epoch: 0004 cost = 0.903608
    Epoch: 0005 cost = 0.554189
    Epoch: 0006 cost = 0.566350
    Epoch: 0007 cost = 0.260842
    Epoch: 0008 cost = 0.279746
    Epoch: 0009 cost = 0.104881
    Epoch: 0010 cost = 0.198070
    Epoch: 0011 cost = 0.183402
    Epoch: 0012 cost = 0.068828
    Epoch: 0013 cost = 0.102589
    Epoch: 0014 cost = 0.238850
    Epoch: 0015 cost = 0.166082
    Epoch: 0016 cost = 0.047024
    Epoch: 0017 cost = 0.046050
    Epoch: 0018 cost = 0.052882
    Epoch: 0019 cost = 0.046769
    Epoch: 0020 cost = 0.022461
    Epoch: 0021 cost = 0.050879
    Epoch: 0022 cost = 0.041592
    Epoch: 0023 cost = 0.016363
    Epoch: 0024 cost = 0.020481
    Epoch: 0025 cost = 0.006086
    Epoch: 0026 cost = 0.015925
    Epoch: 0027 cost = 0.006028
    Epoch: 0028 cost = 0.005126
    Epoch: 0029 cost = 0.004076
    Epoch: 0030 cost = 0.017553
    Epoch: 0031 cost = 0.003573
    Epoch: 0032 cost = 0.002768
    Epoch: 0033 cost = 0.006476
    Epoch: 0034 cost = 0.002022
    Epoch: 0035 cost = 0.004211
    Epoch: 0036 cost = 0.000886
    Epoch: 0037 cost = 0.000744
    Epoch: 0038 cost = 0.001601
    Epoch: 0039 cost = 0.004538
    Epoch: 0040 cost = 0.007894
    Epoch: 0041 cost = 0.000584
    Epoch: 0042 cost = 0.007563
    Epoch: 0043 cost = 0.000536
    Epoch: 0044 cost = 0.002930
    Epoch: 0045 cost = 0.007372
    Epoch: 0046 cost = 0.000542
    Epoch: 0047 cost = 0.000357
    Epoch: 0048 cost = 0.001721
    Epoch: 0049 cost = 0.000792
    Epoch: 0050 cost = 0.000261
    Epoch: 0051 cost = 0.000357
    Epoch: 0052 cost = 0.000756
    Epoch: 0053 cost = 0.001340
    Epoch: 0054 cost = 0.000951
    Epoch: 0055 cost = 0.001984
    Epoch: 0056 cost = 0.000551
    Epoch: 0057 cost = 0.000342
    Epoch: 0058 cost = 0.000688
    Epoch: 0059 cost = 0.001118
    Epoch: 0060 cost = 0.000484
    Epoch: 0061 cost = 0.000617
    Epoch: 0062 cost = 0.000305
    Epoch: 0063 cost = 0.000622
    Epoch: 0064 cost = 0.001073
    Epoch: 0065 cost = 0.000521
    Epoch: 0066 cost = 0.000562
    Epoch: 0067 cost = 0.000814
    Epoch: 0068 cost = 0.000212
    Epoch: 0069 cost = 0.000974
    Epoch: 0070 cost = 0.000524
    Epoch: 0071 cost = 0.000582
    Epoch: 0072 cost = 0.000828
    Epoch: 0073 cost = 0.000226
    Epoch: 0074 cost = 0.000597
    Epoch: 0075 cost = 0.000756
    Epoch: 0076 cost = 0.000319
    Epoch: 0077 cost = 0.000647
    Epoch: 0078 cost = 0.000244
    Epoch: 0079 cost = 0.001193
    Epoch: 0080 cost = 0.001621
    Epoch: 0081 cost = 0.000676
    Epoch: 0082 cost = 0.000316
    Epoch: 0083 cost = 0.000600
    Epoch: 0084 cost = 0.000275
    Epoch: 0085 cost = 0.000190
    Epoch: 0086 cost = 0.000656
    Epoch: 0087 cost = 0.000157
    Epoch: 0088 cost = 0.001317
    Epoch: 0089 cost = 0.000089
    Epoch: 0090 cost = 0.000257
    Epoch: 0091 cost = 0.000325
    Epoch: 0092 cost = 0.000557
    Epoch: 0093 cost = 0.000425
    Epoch: 0094 cost = 0.000264
    Epoch: 0095 cost = 0.000149
    Epoch: 0096 cost = 0.001288
    Epoch: 0097 cost = 0.000196
    Epoch: 0098 cost = 0.000834
    Epoch: 0099 cost = 0.000171
    Epoch: 0100 cost = 0.000340
    최적화 완료!
    

#### 예측함수작성

#### 1
- 입력값과 출력값 데이터로 [영어 단어, 한글 단어]를 사용하지만, 예측시에는 한글 단어를 알지 못한다.

- 따라서 디코더의 입출력을 의미없는 값이 'P'로 채워 데이터를 구성

- seq_data: ['word',PPPP']

- input_batch : ['w','o','l','d']

- output_batch : ['P','P','P','P'] 글자들의 인덱스를 원-핫 인코딩한 값

- target_batch : [2,2,2,2] (['P','P','P','P'], 각 글자의 인덱스)

#### 2

- 결과가 [batch size, time step, input] 으로 나오기 때문에 세번째 차원을 `argmax`로 취해 가장 확률이 높은 글자의 인덱스를 예측값으로 만든다.

#### 3

- 결과 값인 숫자의 인덱스에 해당하는 글자를 가져와 글자 배열을 만든다.

#### 4

- 출력의 끝을 의미하는 'E' 이후의 글자들을 제거하고 문자열로 만든다.

- 디코더의 입력(time steps)크기 만큼 출력값이 나오므로 최종결과는 ['사','랑','E','E']로 나오기 때문 

- **책에 잘못나왔다** ['사', '랑', 'E', 'E', '스'] 으로 나옴

#### tf.argmax(model,2)

- model 의 3번째 차원을 축소함(0,1,2 에서 3번쨰)

- 차원 축소시에는 그 차원의 가장 최대값의 인덱스를 남긴다.


```python
def translate(word):
    print('===============translate===============')
    print(word)
    print('=======================================')
    # 1
    seq_data = [word, 'P' * len(word)]
    input_batch, output_batch, target_batch = make_batch([seq_data])
    
    # 2   
    prediction = tf.argmax(model, 2)
    result = sess.run(prediction, feed_dict={enc_input: input_batch, dec_input: output_batch, targets: target_batch})
    print("result")
    print(result)
    
    # 3
    decoded = [char_arr[i] for i in result[0]]
    print("decode")
    print(decoded)
    
    # 4
    end = decoded.index('E')
    translated = ''.join(decoded[:end])

    return translated
```

#### 단어 번역 테스트 코드 작성


```python
print('=== 번역 테스트 ===')

print('word ->', translate('word'))
print('wodr ->', translate('wodr'))
print('love ->', translate('love'))
print('loev ->', translate('loev'))
print('abcd ->', translate('abcd'))
```

    === 번역 테스트 ===
    ===============translate===============
    word
    =======================================
    input:
    [25, 17, 20, 6]
    output:
    [0, 2, 2, 2, 2]
    target: 
    [2, 2, 2, 2, 1]
    target_batch: 
    [[2, 2, 2, 2, 1]]
    result
    [[29 30  1 34 30]]
    decode
    ['단', '어', 'E', '이', '어']
    word -> 단어
    ===============translate===============
    wodr
    =======================================
    input:
    [25, 17, 6, 20]
    output:
    [0, 2, 2, 2, 2]
    target: 
    [2, 2, 2, 2, 1]
    target_batch: 
    [[2, 2, 2, 2, 1]]
    result
    [[31 32  1 29 30]]
    decode
    ['나', '무', 'E', '단', '어']
    wodr -> 나무
    ===============translate===============
    love
    =======================================
    input:
    [14, 17, 24, 7]
    output:
    [0, 2, 2, 2, 2]
    target: 
    [2, 2, 2, 2, 1]
    target_batch: 
    [[2, 2, 2, 2, 1]]
    result
    [[39 40  1 40 30]]
    decode
    ['사', '랑', 'E', '랑', '어']
    love -> 사랑
    ===============translate===============
    loev
    =======================================
    input:
    [14, 17, 7, 24]
    output:
    [0, 2, 2, 2, 2]
    target: 
    [2, 2, 2, 2, 1]
    target_batch: 
    [[2, 2, 2, 2, 1]]
    result
    [[39 40  1 32 30]]
    decode
    ['사', '랑', 'E', '무', '어']
    loev -> 사랑
    ===============translate===============
    abcd
    =======================================
    input:
    [3, 4, 5, 6]
    output:
    [0, 2, 2, 2, 2]
    target: 
    [2, 2, 2, 2, 1]
    target_batch: 
    [[2, 2, 2, 2, 1]]
    result
    [[21 38 38  1  1]]
    decode
    ['s', '스', '스', 'E', 'E']
    abcd -> s스스
    

#### 결과

- 완전히 상관없는 단어도 그럴듯한 결과를 추측

- 오타를 섞은 다어들도 그럴듯하게 번역
