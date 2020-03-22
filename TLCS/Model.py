import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'  # kill warning about tensorflow
import tensorflow as tf

class Model:
    def __init__(self, image_shape, num_states, num_actions, batch_size):
        self._num_states = num_states
        self.image_shape = image_shape
        self._num_actions = num_actions
        self._batch_size = batch_size

        # define the placeholders
        self._states = None
        self._actions = None

        # the output operations
        self._logits = None
        self._optimizer = None
        self._var_init = None

        # now setup the model
        self._define_model()

    # DEFINE THE STRUCTURE OF THE NEURAL NETWORK
    def _define_model(self):
        # placeholders
        self._image = tf.compat.v1.placeholder(shape=[None, self.image_shape[0],self.image_shape[1],self.image_shape[2]], dtype=tf.float32)
        self._q_s_a = tf.compat.v1.placeholder(shape=[None, self._num_actions], dtype=tf.float32)

        # CNN architect (VGG11)
        x = tf.layers.conv2d(self._image,16, (3,3),(1,1),'same', activation=tf.nn.relu,name='conv2d_1')
        x = tf.layers.conv2d(x,16, (3,3),(2,2),'same', activation=tf.nn.relu,name='conv2d_2')
        x = tf.layers.conv2d(x,32, (3,3),(1,1),'same', activation=tf.nn.relu,name='conv2d_3')
        x = tf.layers.conv2d(x,32, (3,3),(2,2),'same', activation=tf.nn.relu,name='conv2d_4')
        x = tf.layers.conv2d(x,64, (3,3),(1,1),'same', activation=tf.nn.relu,name='conv2d_5')
        x = tf.layers.conv2d(x,64, (3,3),(2,2),'same', activation=tf.nn.relu,name='conv2d_6') #rebuild model
        x = tf.layers.conv2d(x,128, (3,3),(1,1),'same', activation=tf.nn.relu,name='conv2d_7')
        x = tf.layers.conv2d(x,128, (3,3),(2,2),'same', activation=tf.nn.relu,name='conv2d_8')
        x = tf.layers.flatten(x,name='flatten')
        fc1 = tf.layers.dense(x, 512, activation=tf.nn.relu)
        self._logits = tf.layers.dense(fc1, self._num_actions)

        # parameters
        loss = tf.losses.mean_squared_error(self._q_s_a, self._logits)
        self._optimizer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss)
        self._var_init = tf.global_variables_initializer()

    # RETURNS THE OUTPUT OF THE NETWORK GIVEN A SINGLE STATE
    def predict_one(self, image, sess):
        return sess.run(self._logits, feed_dict={self._image: image})

    # RETURNS THE OUTPUT OF THE NETWORK GIVEN A BATCH OF STATES
    def predict_batch(self, images, sess):
        return sess.run(self._logits, feed_dict={self._image: images})

    # TRAIN THE NETWORK
    def train_batch(self, sess, x_batch, y_batch):
        sess.run(self._optimizer, feed_dict={self._image: x_batch, self._q_s_a: y_batch})

    @property
    def num_states(self):
        return self._num_states

    @property
    def num_actions(self):
        return self._num_actions

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def var_init(self):
        return self._var_init
