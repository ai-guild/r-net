import tensorflow as tf
from tensorflow.contrib import rnn

#Gated recurrent units
def gru(num_units):
    gru_cell = rnn.GRUCell(num_units)
    return gru_cell

#Create a stacked gated recurrent units, with n-layer
def gru_n(num_units, num_layers):
    stacked_gru_cells = rnn.MultiRNNCell(
                            [gru(num_units) for _ in range(num_layers)])
    return stacked_gru_cells


def get_variables(n, shape, name='W'):
    return (tf.get_variable(name+str(i), dtype=tf.float32, shape=shape)
               for i in range(n))


'''
    Uni-directional RNN

    [usage]
    cell_ = gru_n(hdim, 3)
    outputs, states = uni_net(cell = cell_,
                             inputs= inputs_emb,
                             init_state= cell_.zero_state(batch_size, tf.float32),
                             timesteps = L)
'''
def uni_net(cell, inputs, init_state, timesteps, time_major=False, scope='uni_net_0'):
    # convert to time major format
    if not time_major:
        inputs_tm = tf.transpose(inputs, [1, 0, -1],name="input_time_major")
    # collection of states and outputs
    states, outputs = [init_state], []

    with tf.variable_scope(scope):

        for i in range(timesteps):
            if i > 0:
                tf.get_variable_scope().reuse_variables()
            output, state = cell(inputs_tm[i], states[-1])
            outputs.append(output)
            states.append(state)

    return tf.stack(outputs), tf.stack(states[1:])


'''
    Bi-directional RNN

    [usage]
    (states_f, states_b), outputs = bi_net(cell_f= gru_n(hdim,3),
                                        cell_b= gru_n(hdim,3),
                                        inputs= inputs_emb,
                                        batch_size= batch_size,
                                        timesteps=L,
                                        scope='bi_net_5',
                                        num_layers=3,
                                        project_outputs=True)
'''
def bi_net(cell_f, cell_b, inputs, batch_size, timesteps, 
           scope= 'bi_net',
           project_outputs=False,
           num_layers=1, hidden_dim = 1):

    # forward
    _, states_f = uni_net(cell_f, 
                          inputs,
                          cell_f.zero_state(batch_size, tf.float32),
                          timesteps,
                          scope=scope + '_f')
    # backward
    _, states_b = uni_net(cell_b, 
                          tf.reverse(inputs, axis=[1]),
                          cell_b.zero_state(batch_size, tf.float32),
                          timesteps,
                          scope=scope + '_b')
    
    cell_f.state_size[0]

    outputs = None
    # outputs
    if project_outputs:
        #chain both forward and backword states together
        states = tf.concat([states_f, states_b], axis=-1)
        
        if len(states.shape) == 4 and num_layers:
            states = tf.reshape(tf.transpose(states, [-2, 0, 1, -1]), [-1, hidden_dim*2*num_layers])
            Wo = tf.get_variable(scope+'/Wo', dtype=tf.float32, shape=[num_layers*2*hidden_dim, hidden_dim])
        elif len(states.shape) == 3:
            states = tf.reshape(tf.transpose(states, [-2, 0, -1]), [-1, hidden_dim*2])
            Wo = tf.get_variable(scope+'/Wo', dtype=tf.float32, shape=[2*hidden_dim, hidden_dim])
        else:
            print('>> ERR : Unable to handle state reshape')
            return None
        
        outputs = tf.reshape(tf.matmul(states, Wo), [batch_size, timesteps, hidden_dim])

    return (states_f, states_b), outputs


'''
    Attention Mechanism

    based on "Neural Machine Translation by Jointly Learning to Align and Translate"
        https://arxiv.org/abs/1409.0473

    [usage]
    ci = attention(enc_states, dec_state, params= {
        'Wa' : Wa, # [d,d]
        'Ua' : Ua, # [d,d]
        'Va' : Va  # [d,1]
        })
    shape(enc_states) : [B, L, d]
    shape(dec_state)  : [B, d]
    shape(ci)         : [B,d]

'''
def attention(enc_states, dec_state, params, d, timesteps):
    Wa, Ua = params['Wa'], params['Ua']
    # s_ij -> [B,L,d]
    a = tf.tanh(tf.expand_dims(tf.matmul(dec_state, Wa), axis=1) + 
            tf.reshape(tf.matmul(tf.reshape(enc_states,[-1, d]), Ua), [-1, timesteps, d]))
    Va = params['Va'] # [d, 1]
    # e_ij -> softmax(aV_a) : [B, L]
    scores = tf.nn.softmax(tf.reshape(tf.matmul(tf.reshape(a, [-1, d]), Va), [-1, timesteps]))
    # c_i -> weighted sum of encoder states
    return tf.reduce_sum(enc_states*tf.expand_dims(scores, axis=-1), axis=1) # [B, d]    


'''
    Attentive Decoder

    [usage]
    dec_outputs, dec_states = attentive_decoder(enc_states,
                                    tf.zeros(dtype=tf.float32, shape=[B,d]),
                                    batch_size=B,timesteps=L,feed_previous=True,
                                    inputs = inputs)
    shape(enc_states) : [B, L, d]
    shape(inputs) : [[B, d]] if feed_previous else [L, B, d]


'''
def attentive_decoder(enc_states, init_state, batch_size, 
                      d, timesteps,
                      inputs = [],
                      scope='attentive_decoder_0',
                      num_layers=1,
                      feed_previous=False):
    # get parameters
    U,W,C,Ur,Wr,Cr,Uz,Wz,Cz,Uo,Vo,Co = get_variables(12, [d,d], name='decoder_param')
    Wa, Ua = get_variables(2, [d,d], 'att')
    Va = tf.get_variable('Va', shape=[d, 1], dtype=tf.float32)
    att_params = {
        'Wa' : Wa, 'Ua' : Ua, 'Va' : Va
    }
    
        
    def step(input_, state, ci):
        z = tf.nn.sigmoid(tf.matmul(input_, Wz)+tf.matmul(state, Uz)+tf.matmul(ci, Cz))
        r = tf.nn.sigmoid(tf.matmul(input_, Wr)+tf.matmul(state, Ur)+tf.matmul(ci, Cr))
        si = tf.nn.tanh(tf.matmul(input_, W)+tf.matmul(ci, C)+tf.matmul(r*state, U))
        
        state = (1-z)*state + z*si
        output = tf.matmul(state, Uo) + tf.matmul(input_, Vo) + tf.matmul(ci, Co)
        
        return output, state
    
    outputs = [inputs[0]] # include GO token as init input
    states = [init_state]
    for i in range(timesteps):
        input_ = outputs[-1] if feed_previous else inputs[i]
        output, state = step(input_, states[-1],
                            attention(enc_states, states[-1], att_params, d, timesteps))
    
        outputs.append(output)
        states.append(state)
    # time major -> batch major
    states_bm = tf.transpose(tf.stack(states[1:]), [1, 0, 2])
    outputs_bm = tf.transpose(tf.stack(outputs[1:]), [1, 0, 2])
    return outputs_bm, states_bm


'''
    Gated Attention Network

    based on "R-NET: Machine Reading Comprehension with Self-matching Networks"
        https://www.microsoft.com/en-us/research/publication/mrc/

    [usage]
    dec_outputs, dec_states = gated_attention_net(enc_states, # encoded representation of text
                                    tf.zeros(dtype=tf.float32, shape=[B,d*2]), # notice d*2
                                    batch_size=B,timesteps=L,feed_previous=False,
                                    inputs = inputs)
    shape(enc_states) : [B, L, d]
    shape(inputs) : [[B, d]] if feed_previous else [L, B, d]

    For reading comprehension, inputs is same as enc_states; feed_previous doesn't apply


'''
def gated_attention_net(states_a, states_b, init_state_c, 
                        batch_size, d, La, Lb, 
                        scope='gated_attention_net_0'):
    
    with tf.variable_scope(scope, reuse=False):
        # define attention parameters
        Wa = tf.get_variable('Wa', shape=[d, d], dtype=tf.float32)
        Wb = tf.get_variable('Wb', shape=[d, d], dtype=tf.float32)
        Wc = tf.get_variable('Wc', shape=[d, d], dtype=tf.float32)
        Va = tf.get_variable('Va', shape=[d, 1], dtype=tf.float32)

        att_params = {
            'Wa' : Wa, 'Wb' : Wb, 'Wc' : Wc, 'Va' : Va
        }

        # input gate/projection
        Wg = tf.get_variable('Wg', shape=[d, d], dtype=tf.float32)
        Wi = tf.get_variable('Wi', shape=[d*2, d], dtype=tf.float32)

    # define rnn cell
    cell = gru(num_units=d)
    
    # convert states_a to batch_major
    states_a = tf.transpose(states_a, [1,0,2])
        
    def step(input_, state):
        # define input gate
        gi = tf.nn.sigmoid(tf.matmul(input_, Wg))
        # apply gate to input
        input_ = gi * input_
        # recurrent step
        output, state = cell(input_, state)
        return output, state
    
    states = [init_state_c]
    for i in range(Lb):
        if i>0:
            tf.get_variable_scope().reuse_variables()

        # get match for current word
        ci = attention_pooling(states_a, states_b[i], states[-1], 
                                att_params, d, La)
        # combine ci and input(i) 
        input_ = tf.matmul(tf.concat([states_b[i], ci], axis=-1), Wi)
        _, state = step(input_, states[-1])
    
        states.append(state)

    # time major -> batch major
    states_bm = tf.transpose(tf.stack(states[1:]), [1, 0, 2])
    #outputs_bm = tf.transpose(tf.stack(outputs[1:]), [1, 0, 2])

    return states_bm

'''
    Attention Pooling Mechanism

    based on "R-NET: Machine Reading Comprehension with Self-matching Networks"
        https://www.microsoft.com/en-us/research/publication/mrc/

    [usage]
    ci = attention(qstates, pstates_i, state, params= {
        'Wa' : Wa, # [d,d]
        'Wb' : Wb, # [d,d]
        'Wc' : Wc, # [d,d]
        'Va' : Va  # [d,1]
        })
    shape(qstates) : [B, L, d]
    shape(pstates_i) : [B, d]
    shape(state)     : [B, d]
    shape(ci)        : [B, d]

'''
def attention_pooling(states_a, states_b_i, state_c, params, d, timesteps):
    Wa, Wb, Wc = params['Wa'], params['Wb'], params['Wc']
    # s_ij -> [B,L,d]
    a = tf.tanh(tf.expand_dims(tf.matmul(states_b_i, Wb), axis=1) +
            tf.reshape(tf.matmul(tf.reshape(states_a,[-1, d]), Wa), [-1, timesteps, d]) +
            tf.expand_dims(tf.matmul(state_c, Wc), axis=1))
    Va = params['Va'] # [d, 1]
    # e_ij -> softmax(aV_a) : [B, L]
    scores = tf.nn.softmax(tf.reshape(tf.matmul(tf.reshape(a, [-1, d]), Va), [-1, timesteps]))
    # c_i -> weighted sum of encoder states
    return tf.reduce_sum(states_a*tf.expand_dims(scores, axis=-1), axis=1) # [B, d]

