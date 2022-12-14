import os
import json

import tensorflow as tf
import mnist


per_worker_batch_size = 64

job_id = os.environ['TF_JOB_ID']

cluster = [
    ('cuda10', '17000'),
    ('cuda10', '17001'),
    ('cuda10', '17002'),
    ('cuda1', '17000'),
    ('cuda1', '17001'),
]

# set current worker bind address to 0.0.0.0
workers = []
for i, (ip, port) in enumerate(cluster):
    if int(job_id) == i:
        ip = '0.0.0.0'
    workers.append(f"{ip}:{port}")

tf_config = {
    'cluster': {
        'worker': workers,
    },
    'task': {'type': 'worker', 'index': job_id }
}

os.environ['TF_CONFIG'] = json.dumps(tf_config)
num_workers = len(tf_config['cluster']['worker'])

configs_to_print = ['TF_JOB_ID', 'TF_CONFIG']
for c in configs_to_print:
    print(c, ':', os.environ[c])

strategy = tf.distribute.MultiWorkerMirroredStrategy()

global_batch_size = per_worker_batch_size * num_workers
multi_worker_dataset = mnist.mnist_dataset(global_batch_size)

with strategy.scope():
    # Model building/compiling need to be within `strategy.scope()`.
    multi_worker_model = mnist.build_and_compile_cnn_model()


multi_worker_model.fit(multi_worker_dataset, epochs=3, steps_per_epoch=70)
