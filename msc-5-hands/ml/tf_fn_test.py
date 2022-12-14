import tensorflow as tf

def make_tarray_fns():

    def create_state():
        return tf.TensorArray(tf.float32, size=1, dynamic_size=False, infer_shape=False, element_shape=[None, 2])

    def irmqa(state):

        return state.write(state.size(), tf.random.uniform([1, 2]))
    
    def loop(state):

        for _ in range(10):
            state = irmqa(state)

        return state

    return create_state, loop


def make_tensor_fns():

    def create_state():
        return tf.zeros([0, 2], dtype=tf.float32)

    def irmqa(state):

        return tf.concat([state, tf.random.uniform([1, 2])], axis=0)
    
    def loop(state):

        for _ in range(10):
            state = irmqa(state)

        return state

    return create_state, loop

# create_state, loop = make_tarray_fns()
# fn = tf.function(loop, input_signature=[tf.TensorArraySpec(element_shape=[None, 2], dtype=tf.float32, dynamic_size=True)])
create_state, loop = make_tensor_fns()
fn = tf.function(loop)
state = create_state()
state = fn(state)
state = fn(state)

tf.print(state)
