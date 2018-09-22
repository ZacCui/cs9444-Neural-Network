import tensorflow as tf
import re

BATCH_SIZE = 128
MAX_WORDS_IN_REVIEW = 200  # Maximum length of a review to consider
EMBEDDING_SIZE = 50  # Dimensions for each word vector
num_units = 32
learning_rate = 0.001

stop_words = set({'ourselves', 'hers', 'between', 'yourself', 'again',
                  'there', 'about', 'once', 'during', 'out', 'very', 'having',
                  'with', 'they', 'own', 'an', 'be', 'some', 'for', 'do', 'its',
                  'yours', 'such', 'into', 'of', 'most', 'itself', 'other',
                  'off', 'is', 's', 'am', 'or', 'who', 'as', 'from', 'him',
                  'each', 'the', 'themselves', 'below', 'are', 'we',
                  'these', 'your', 'his', 'through', 'don', 'me', 'were',
                  'her', 'more', 'himself', 'this', 'down', 'should', 'our',
                  'their', 'while', 'above', 'both', 'up', 'to', 'ours', 'had',
                  'she', 'all', 'no', 'when', 'at', 'any', 'before', 'them',
                  'same', 'and', 'been', 'have', 'in', 'will', 'on', 'does',
                  'yourselves', 'then', 'that', 'because', 'what', 'over',
                  'why', 'so', 'can', 'did', 'not', 'now', 'under', 'he', 'you',
                  'herself', 'has', 'just', 'where', 'too', 'only', 'myself',
                  'which', 'those', 'i', 'after', 'few', 'whom', 't', 'being',
                  'if', 'theirs', 'my', 'against', 'a', 'by', 'doing', 'it',
                  'how', 'further', 'was', 'here', 'than', 'nor', 'must', 
                  'doesn', 'haven', 'wouldn', 'shouldn', 'onto'})

def preprocess(review):
    """
    Apply preprocessing to a single review. You can do anything here that is manipulation
    at a string level, e.g.
        - removing stop words
        - stripping/adding punctuation
        - changing case
        - word find/replace
    RETURN: the preprocessed review in string form.
    """
    processed_review = []

    # convert review to lower case
    review = review.lower()

    # remove <br /><br />
    review = re.sub("<br /><br />", " ", review)

    # remove punctuation, string of all punctuation: !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~
    '''
    punctuation = string.punctuation
    for character in review:
        if character not in punctuation:
            tmp_review += character
    '''
    tmp_review = ""
    tmp_review = re.sub(r"[\!\"\#\$\%\&\'\(\)\*\+\,\-\.\/\:\;\<\=\>\?\@\[\]\^\_\`\{\|\}\~]", " ", review)

    # delete multiple space
    tmp_review = re.sub(r"\s+", " ", tmp_review)

    # check number of words, removing stop words and other meaningless single letter
    words_arr = tmp_review.split(" ")
    length = 0
    for word in words_arr:
        if length >= MAX_WORDS_IN_REVIEW:
            break
        if word not in stop_words and len(word) > 1:
            processed_review.append(word)
            length += 1


    return processed_review



def define_graph():
    """
    Implement your model here. You will need to define placeholders, for the input and labels,
    Note that the input is not strings of words, but the strings after the embedding lookup
    has been applied (i.e. arrays of floats).

    In all cases this code will be called by an unaltered runner.py. You should read this
    file and ensure your code here is compatible.

    Consult the assignment specification for details of which parts of the TF API are
    permitted for use in this function.

    You must return, in the following order, the placeholders/tensors for;
    RETURNS: input, labels, optimizer, accuracy and loss
    """
    input_data = tf.placeholder(dtype=tf.float32, shape=[BATCH_SIZE, MAX_WORDS_IN_REVIEW, EMBEDDING_SIZE], name="input_data")
    labels = tf.placeholder(dtype=tf.float32, shape=[BATCH_SIZE, 2], name="labels")
    dropout_keep_prob = tf.placeholder_with_default(0.6, shape=(), name="dropout_keep_prob")
    number_of_layers = 2

    model = 0
    batch_output = False
    if (model == 1):
        multi_lstm = tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in range(number_of_layers)])
        multi_lstm = tf.contrib.rnn.DropoutWrapper(multi_lstm, output_keep_prob=dropout_keep_prob)
        (outputs, state) = tf.nn.dynamic_rnn(cell=multi_lstm, inputs=input_data, dtype=tf.float32)
    else:
        lstm_fw_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell(), output_keep_prob=dropout_keep_prob)
        lstm_bw_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell(), output_keep_prob=dropout_keep_prob)
        (outputs, output_state_fw, output_state_bw) = tf.contrib.rnn.stack_bidirectional_dynamic_rnn([lstm_fw_cell], [lstm_bw_cell], inputs=input_data, dtype=tf.float32)

    if batch_output:
        outputs = tf.reduce_mean(outputs, axis=1)
    
    w = tf.Variable(tf.random_normal([num_units * 2, 2]), dtype=tf.float32)
    b = tf.Variable(tf.random_normal([2]), dtype=tf.float32)
    logits = tf.add(tf.matmul(outputs[:, -1, :], w), b)
    xentropy = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
    loss = tf.reduce_mean(xentropy, name="loss")

    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    correct_preds = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))

    Accuracy = tf.reduce_mean(tf.cast(correct_preds, tf.float32), name="accuracy")

    return input_data, labels, dropout_keep_prob, optimizer, Accuracy, loss


def lstm_cell():
    return tf.contrib.rnn.BasicLSTMCell(num_units)