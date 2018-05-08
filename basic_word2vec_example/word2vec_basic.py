import math
import numpy as np
import tensorflow as tf
import os
from data_utils import maybe_download, read_data, build_dataset, tsne_and_plot, generate_batch

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # suppress tensorflow warnings

# Step 1: Download the data (maybe).
filename = maybe_download('text8.zip', 31344016)

# Step 2: Read and build the dictionary and replace rare words with UNK token.
vocabulary = read_data(filename)
print('Data size', len(vocabulary))
vocabulary_size = 50000
data, count, dictionary, reverse_dictionary = build_dataset(vocabulary, vocabulary_size)
del vocabulary  # Hint to reduce memory.
print('Most common words (+UNK)', count[:5])
print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])


# Step 3: get batch data generator
data_index = 0
sample_batch, sample_labels, data_index = generate_batch(data, data_index, batch_sz=8, n_skips=2, skip_wd=1)
for i in range(8):
    print(sample_batch[i], reverse_dictionary[sample_batch[i]], '->', sample_labels[i, 0],
          reverse_dictionary[sample_labels[i, 0]])


# Step 4: Build a skip-gram model.
batch_size = 128
embedding_size = 128  # Dimension of the embedding vector.
skip_window = 1  # How many words to consider left and right.
num_skips = 2   # How many times to reuse an input to generate a label.
num_sampled = 64  # Number of negative examples to sample.
# We pick a random validation set to sample nearest neighbors. Here we limit the validation samples to the words that
# have a low numeric ID, which by construction are also the most frequent. These 3 variables are used only for
# displaying model accuracy, they don't affect calculation.
valid_size = 16  # Random set of words to evaluate similarity on.
valid_window = 100  # Only pick dev samples in the head of the distribution.
valid_examples = np.random.choice(valid_window, valid_size, replace=False)

graph = tf.Graph()
with graph.as_default():
    # Input data.
    train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
    train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)
    # Ops and variables pinned to the CPU because of missing GPU implementation
    with tf.device('/cpu:0'):
        # Look up embeddings for inputs.
        embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
        embed = tf.nn.embedding_lookup(embeddings, train_inputs)
        # Construct the variables for the NCE loss
        nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size],
                                                      stddev=1.0 / math.sqrt(embedding_size)))
        nce_biases = tf.Variable(tf.zeros([vocabulary_size]))
    # Compute the average NCE loss for the batch.
    # tf.nce_loss automatically draws a new sample of the negative labels each time we evaluate the loss.
    loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weights, biases=nce_biases, labels=train_labels, inputs=embed,
                                         num_sampled=num_sampled, num_classes=vocabulary_size))
    # Construct the SGD optimizer using a learning rate of 1.0.
    optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)
    # Compute the cosine similarity between minibatch examples and all embeddings.
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
    normalized_embeddings = tf.divide(embeddings, norm)
    # normalized_embeddings = embeddings / norm
    valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
    similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)

# Step 5: Begin training.
num_steps = 100001
with tf.Session(graph=graph) as sess:
    sess.run(tf.global_variables_initializer())  # initialize all variables
    average_loss = 0
    for step in range(num_steps):
        batch_inputs, batch_labels, data_index = generate_batch(data, data_index, batch_size, num_skips, skip_window)
        # perform one update step by evaluating the optimizer(including it in the list of returned values for sess.run()
        _, loss_val = sess.run([optimizer, loss], feed_dict={train_inputs: batch_inputs, train_labels: batch_labels})
        average_loss += loss_val

        if step % 2000 == 0:
            if step > 0:
                average_loss /= 2000
            # The average loss is an estimate of the loss over the last 2000 batches.
            print('Average loss at step ', step, ': ', average_loss)
            average_loss = 0

        # Note that this is expensive (~20% slowdown if computed every 500 steps)
        if step % 10000 == 0:
            sim = similarity.eval()
            for i in range(valid_size):
                valid_word = reverse_dictionary[valid_examples[i]]
                top_k = 8  # number of nearest neighbors
                nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                log_str = 'Nearest to %s:' % valid_word
                for k in range(top_k):
                    close_word = reverse_dictionary[nearest[k]]
                    log_str = '%s %s,' % (log_str, close_word)
                print(log_str)
    final_embeddings = normalized_embeddings.eval()

# Step 6: Visualize the embeddings.,
plot_only = 500
embeddings = final_embeddings[:plot_only, :]
sample_labels = [reverse_dictionary[i] for i in range(plot_only)]
tsne_and_plot(embeddings, sample_labels, filename='tsne.png')
