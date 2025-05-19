# from _future_ import print_function
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# print('tensorflow version:', tf._version_)

# --- Define Transformer Components ---

class TransformerEncoderLayer(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads=4, ff_dim=None, dropout_rate=0.1):
        super(TransformerEncoderLayer, self).__init__()
        if ff_dim is None:
            ff_dim = embed_dim * 2
        self.attention = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential([
            layers.Dense(ff_dim, activation=tf.keras.activations.gelu),
            layers.Dense(embed_dim)
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)
    
    def call(self, inputs, training=False, mask=None):
        attn_output = self.attention(inputs, inputs, inputs, attention_mask=mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

class TransformerEncoder(tf.keras.Model):
    def __init__(self, num_layers, embed_dim, num_heads=4, ff_dim=None, dropout_rate=0.1):
        super(TransformerEncoder, self).__init__()
        self.num_layers = num_layers
        self.enc_layers = [TransformerEncoderLayer(embed_dim, num_heads, ff_dim, dropout_rate)
                           for _ in range(num_layers)]
        self.flatten = layers.Flatten()
        self.dense = layers.Dense(64, activation='relu')
        self.out_dense = layers.Dense(0, activation='sigmoid')  # placeholder; will be replaced later

    def call(self, inputs, training=False, mask=None):
        x = inputs
        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)
        x = self.flatten(x)
        x = self.dense(x)
        return x

# --- MemoryDNN using Transformer Encoder ---
class MemoryDNN:
    def __init__(self, net, learning_rate=0.01, training_interval=10, batch_size=100, memory_size=1000, output_graph=False):
        # net should be [input_dim, hidden, ..., output_dim]; here input_dim = N*3, output_dim = N.
        self.net = net  
        self.training_interval = training_interval
        self.lr = learning_rate
        self.batch_size = batch_size
        self.memory_size = memory_size
        self.enumerate_actions = []
        self.memory_counter = 1
        self.cost_his = []
        self.memory = np.zeros((self.memory_size, self.net[0] + self.net[-1]))
        self._build_net()

    def _build_net(self):
        # For the transformer model, we interpret the input as a sequence of N tokens, each of dimension 3.
        num_tokens = int(self.net[0] / 3)
        embed_dim = 128
        num_layers = 2  # You can adjust the number of transformer layers as needed.
        
        inputs = keras.Input(shape=(num_tokens, 3))
        # A simple linear projection to embedding dimension.
        x = layers.Dense(embed_dim)(inputs)
        # Add positional encoding (optional, here we simply add a learned embedding)
        pos_emb = layers.Embedding(input_dim=num_tokens, output_dim=embed_dim)
        positions = tf.range(start=0, limit=num_tokens, delta=1)
        pos_encoding = pos_emb(positions)
        x = x + pos_encoding
        # Transformer encoder layers
        encoder_layer = TransformerEncoder(num_layers=num_layers,embed_dim=embed_dim)
        x = encoder_layer(x)
        x = layers.Flatten()(x)
        x = layers.Dense(64, activation='relu')(x)
        outputs = layers.Dense(self.net[-1], activation='sigmoid')(x)
        self.model = keras.Model(inputs=inputs, outputs=outputs)
        self.model.compile(optimizer=keras.optimizers.Adam(learning_rate=self.lr),
                           loss=tf.keras.losses.BinaryCrossentropy(),
                           metrics=['accuracy'])
        self.model.summary()

    def remember(self, h, m):
        idx = self.memory_counter % self.memory_size
        self.memory[idx, :] = np.hstack((h, m))
        self.memory_counter += 1

    def encode(self, h, m):
        self.remember(h, m)
        if self.memory_counter % self.training_interval == 0:
            self.learn()

    def learn(self):
        if self.memory_counter < self.batch_size:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size, replace=True)
        elif self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size, replace=False)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size, replace=False)
        batch_memory = self.memory[sample_index, :]
        input_dim = self.net[0]
        # Reshape training inputs to (batch_size, num_tokens, 3)
        h_train = batch_memory[:, :input_dim].reshape(self.batch_size, int(self.net[0] / 3), 3)
        m_train = batch_memory[:, input_dim:]
        hist = self.model.fit(h_train, m_train, verbose=0)
        loss_val = hist.history['loss'][0]
        assert loss_val >= 0, f"Unexpected negative loss: {loss_val}"
        self.cost_his.append(loss_val)

    def decode(self, h, k=1, mode='OP'):
        # h: numpy array of shape (N*3,)
        num_tokens = int(self.net[0] / 3)
        h = h.reshape(num_tokens, 3)
        h = h[np.newaxis, :]  # shape: (1, num_tokens, 3)
        m_pred = self.model.predict(h, verbose=0)
        m_candidate = (m_pred[0] > 0.5).astype(int)
        return (m_pred[0], [m_candidate])

    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(np.arange(len(self.cost_his)) * self.training_interval, self.cost_his)
        plt.ylabel('Training Loss')
        plt.xlabel('Time Frames')
        plt.title('Transformer-based MemoryDNN Training Cost')
        plt.show()