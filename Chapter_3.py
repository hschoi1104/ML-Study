#!/usr/bin/env python
# coding: utf-8

# ## Chapter 3.1

# In[2]:


import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


# In[4]:


hello = tf.constant('Hello,TensorFlow!')
print(hello)


# In[6]:


sess = tf.Session()
a = tf.constant(10)
b = tf.constant(32)
c = tf.add(a,b)
print(c)


# In[7]:


print(sess.run(hello))
print(sess.run([a,b,c]))

sess.close()


# ## ch3.2
# ### placeholder and variable

# In[8]:


X = tf.placeholder(tf.float32, [None,3])
print(X)


# In[15]:


x_data = [[1,2,3],[4,5,6]]


# In[16]:


W = tf.Variable(tf.random_normal([3,2]))
B = tf.Variable(tf.random_normal([2,1]))
#W = tf.Variable([[0.1,0.1],[0.2,0.2],[0.3,0.3]])


# In[17]:


expr = tf.matmul(X,W)+b
print(expr)


# In[23]:


sess = tf.Session()
sess.run(tf.global_variables_initializer())

print("===x-data===")
print(x_data)
print("===W===")
print(sess.run(W))
print("===B===")
print(sess.run(b))
print("===expr===")
print(sess.run(expr,feed_dict={X:x_data}))

sess.close()


# ## ch3.3
# ### 선형 회귀 모델

# In[25]:


x_data = [1,2,3]
y_data = [1,2,3]


# In[42]:


W = tf.Variable(tf.random_uniform([1],-0.1,1.0))
b = tf.Variable(tf.random_uniform([1],-0.1,1.0))


# In[43]:


X = tf.placeholder(tf.float32, name = "X")
Y = tf.placeholder(tf.float32, name = "Y")


# In[44]:


hypothesis = W * X + b


# In[45]:


cost = tf.reduce_mean(tf.square(hypothesis -Y))


# In[46]:


optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
train_op = optimizer.minimize(cost)


# In[54]:


sess = tf.Session()
sess.run(tf.global_variables_initializer())
for step in range(100):
    _,cost_val = sess.run([train_op, cost], feed_dict={X: x_data, Y: y_data})
    print(step, cost_val,sess.run(W), sess.run(b))


# In[55]:


print("X: 5, Y:",sess.run(hypothesis, feed_dict={X: 5}))
print("X: 2.5, Y:", sess.run(hypothesis, feed_dict={X: 2.5}))
sess.close()


# In[ ]:




