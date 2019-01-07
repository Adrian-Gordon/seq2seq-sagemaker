import tensorflow as tf

import functools


    
def define_scope(function):
    attribute ='_cache_' + function.__name__

    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            with tf.variable_scope(function.__name__):
                setattr(self, attribute, function(self))
        return getattr(self, attribute)
    
    return decorator

class Seq2Seq:
    def __init__(self, config):

        with tf.variable_scope("placeholders"):
          self.encoder_inputs =[
              tf.placeholder(tf.float64, shape =(None, config["input_dim"]), name="input_{}".format(t))
              for t in range(config["input_sequence_length"])
          ]

          self.decoder_target_inputs =[
              tf.placeholder(tf.float64, shape =(None, config["output_dim"]), name="output_{}".format(t))
              for t in range(config["output_sequence_length"])
          ]
    
        self.weights =  tf.get_variable('Weights_out', shape = [config["hidden_dim"], config["output_dim"]], dtype = tf.float64, initializer = tf.truncated_normal_initializer())
        self.biases = tf.get_variable('Biases_out', shape = [config["output_dim"]], dtype = tf.float64, initializer = tf.constant_initializer(0.))
        self.global_step = tf.Variable(
            initial_value=0,
            name="global_step",
            trainable=False,
            collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES])

        self.encoder_decoder
        self.encode_decode
        self.loss
        self.optimize

    #build the encoder/decoder cells
    @define_scope
    def encoder_decoder(self):
        with tf.variable_scope("EncoderDecoderLSTMCell", reuse=tf.AUTO_REUSE):

            encode_cells=[]
            for i in range(config["num_stacked_layers"]):
                with tf.variable_scope('encode_RNN_{}'.format(i)):
                    encode_cells.append(tf.contrib.rnn.LSTMCell(config["hidden_dim"]))
                    
            self.encode_cell = tf.contrib.rnn.MultiRNNCell(encode_cells)

            decode_cells=[]
            for i in range(config["num_stacked_layers"]):
                with tf.variable_scope('decode_RNN_{}'.format(i)):
                    decode_cells.append(tf.contrib.rnn.LSTMCell(config["hidden_dim"]))
            self.decode_cell = tf.contrib.rnn.MultiRNNCell(decode_cells)

        return self.encode_cell, self.decode_cell


    @define_scope
    def encode_decode(self):
        with tf.variable_scope("EncoderDecode", reuse=tf.AUTO_REUSE):
            decoder_outputs=[]
            encode_cell, decode_cell = self.encoder_decoder

            #put a 'GO' token at the start of the target decoder inputs - losing the one at the end

            decoder_inputs = [ tf.zeros_like(self.decoder_target_inputs[0], dtype=tf.float64, name="GO-Train") ] + self.decoder_target_inputs[:-1]

            encoder_outputs, states = tf.contrib.rnn.static_rnn(encode_cell, self.encoder_inputs, dtype=tf.float64)
            #iterate over the decoder inputs
            for i, decoder_input in enumerate(decoder_inputs):
                decoder_output, states = decode_cell(decoder_input,states) #pass in new data and previous state, update the state for the next cycle
                decoder_outputs.append(decoder_output)
            return  [tf.matmul(i, self.weights) + self.biases for i in decoder_outputs]
    @define_scope
    def encoder_decoder_inference(self):
        encode_cell, decode_cell = self.encoder_decoder
        inferred_outputs=[]
        #decoder_input = tf.constant([[0]], dtype=tf.float64) #the 'Go' token
        #all are dummy 'Go' tokens in this case - only the first one is actually used
        decoder_inputs = [ tf.zeros_like(self.decoder_target_inputs[0], dtype=tf.float64, name="GO-Inference") ] + self.decoder_target_inputs[:-1]
        #decoder_input = decoder_input[0]
        #print decoder_input
        encoder_outputs, states = tf.contrib.rnn.static_rnn(encode_cell, self.encoder_inputs, dtype=tf.float64)

        reshaped_decoder_output = None

        for i, decoder_input in enumerate(decoder_inputs):
            if reshaped_decoder_output ==None:
                decoder_output, states = decode_cell(decoder_input, states)         ##pass in 'Go' data, and state, from encoder
                reshaped_decoder_output = tf.matmul(decoder_output, self.weights) + self.biases
            else:
                decoder_output, states = decode_cell(reshaped_decoder_output, states) #pass in previous (generated) data point and previous state - generates an output, and a new decoder state
                reshaped_decoder_output = tf.matmul(decoder_output, self.weights) + self.biases
                #  print(reshaped_decoder_output)
            inferred_outputs.append(reshaped_decoder_output)

        return inferred_outputs

    @define_scope
    def loss(self):
        with tf.variable_scope('Loss'):
            output_loss =  tf.reduce_mean(tf.pow(tf.subtract(self.encode_decode,self.decoder_target_inputs),2))
            regularization_loss = 0

            for tf_var in tf.trainable_variables():
                if 'Biases_' in tf_var.name or 'Weights_' in tf_var.name:
                    regularization_loss += tf.reduce_mean(tf.nn.l2_loss(tf_var)) 

            loss = output_loss + (config["l2_regularization_lambda"] * regularization_loss)
            return loss

      

    @define_scope
    def optimize(self):
        with tf.variable_scope('Optimizer'):
            optimizer = tf.contrib.layers.optimize_loss(
                loss=self.loss,
                learning_rate=config["learning_rate"],
                global_step=self.global_step,
                optimizer='Adam',
                clip_gradients=config["gradient_clipping"])
            return optimizer


#test
import boto3
import json
bucket='culturehub'
configuration_key ='seqtoseq/config_beijing.json'
configuration_location = 's3://{}/{}'.format(bucket, configuration_key)

configuration_string = boto3.resource('s3').Object(bucket, configuration_key).get()['Body'].read().decode('utf-8')

#print(configuration_string)

config = json.loads(configuration_string)

seq2seq  = Seq2Seq(config)