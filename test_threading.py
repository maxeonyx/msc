# coding: utf-8
import threading
import tensorflow as tf
gpu = tf.config.list_physical_devices('GPU')[0]
tf.config.experimental.set_memory_growth(gpu, True)
from tensorflow import keras
from tensorflow.keras.datasets import mnist
data = mnist.load_data()
(train_x, train_y), (test_x, test_y) = data
test_x, test_y, train_x, train_y = [tf.constant(x) for x in [test_x, test_y, train_x, train_y]]

model = keras.Sequential([
    keras.layers.Flatten(),
    keras.layers.Dense(100, activation='relu'),
    keras.layers.Dense(10, activation='softmax'),
])
model.build(input_shape=[None, 28, 28])
def do_train():
    model.compile(loss=keras.losses.SparseCategoricalCrossentropy(), optimizer=keras.optimizers.Adam())
    model.fit(x=train_x, y=train_y, validation_data=[test_x, test_y], epochs=10)

stop_event = threading.Event()    
def print_var():
    import time
    while not stop_event.is_set():
        print("var val:", tf.reduce_sum(model.layers[1].weights[0]).numpy())
        time.sleep(.1)

stop_event.clear()
t1 = threading.Thread(target=do_train)
t2 = threading.Thread(target=print_var)
t1.start()
t2.start()
t1.join()
stop_event.set()
t2.join()
