from sklearn.cluster import MiniBatchKMeans
import os
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import Model, Input, layers
from IPython.display import display
import tensorflow_datasets as tfds
import time
import matplotlib.pyplot as plt
import enlighten
import tensorflow_probability as tfp
from dotmap import DotMap        
import wandb

import models

def training_functions(config, model, optimizer, ds_train):
    
    loss_function = keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
    weights = model.trainable_weights
    
    spec = (
        tf.TensorSpec(shape=(config['minibatch_size'], None), dtype=tf.int32),
        tf.TensorSpec(shape=(config['minibatch_size'], None), dtype=tf.int32),
        tf.TensorSpec(shape=(), dtype=tf.int32),
    )
    if config.distributed:
        dist_spec = ds_train.element_spec
        
        dist_spec[0]._value_specs = tuple([spec[0]]*config.num_devices)
        dist_spec[1]._value_specs = tuple([spec[1]]*config.num_devices)
        dist_spec[2]._value_specs = tuple([spec[2]]*config.num_devices)
        spec = dist_spec
    
    ds_train = iter(ds_train)

    # @tf.function(input_signature=[spec])
    @tf.function
    def train_step_inner(inputs):
        colors, idxs, n = inputs
        n = tf.squeeze(n)
        x_inp = colors[:, :n]
        x_tar = colors[:, n:]
        i_inp = idxs[:, :n]
        i_tar = idxs[:, n:]
        inp_seq_len = tf.shape(i_inp)[-1]
        tar_seq_len = tf.shape(i_tar)[-1]
        enc_a_mask = models.get_mask(models.MASK_NONE, inp_seq_len, inp_seq_len)
        dec_mask = models.get_mask(models.MASK_NONE, inp_seq_len, tar_seq_len)
        with tf.GradientTape() as tape:
            x_out = model([x_inp, i_inp, i_tar, enc_a_mask, dec_mask], training=True)
            loss = loss_function(x_tar, x_out)
            gradients = tape.gradient(loss, weights)
        return loss, gradients
    
    @tf.function
    def train_step_grad_accum(accum_steps):
        float_steps = tf.cast(accum_steps, tf.float32)
        accum_gradients = [tf.zeros_like(w, dtype=tf.float32) for w in weights]
        accum_loss = tf.constant(0, tf.float32)
        for step in tf.range(accum_steps):
            inputs = next(ds_train)
            loss, gradients = train_step_inner(inputs)
            accum_gradients = [accum_grad+grad for accum_grad, grad in zip(accum_gradients, gradients)]
            accum_loss += tf.reduce_mean(loss)
        # uncomment to divide gradient. without dividing, bigger steps (still kind of averaged)
        # accum_gradients = [accum_grad / float_steps for accum_grad in accum_gradients]
        optimizer.apply_gradients(zip(accum_gradients, weights))
        accum_loss /= float_steps
        return accum_loss
    
    # @tf.function(input_signature=[spec])
    @tf.function
    def train_step_normal(inputs):
        loss, gradients = train_step_inner(inputs)
        optimizer.apply_gradients(zip(gradients, weights))
        return loss

    @tf.function(input_signature=[
        tf.TensorSpec(shape=(None, None), dtype=tf.int32),
        tf.TensorSpec(shape=(None, None), dtype=tf.int32),
        tf.TensorSpec(shape=(None, None), dtype=tf.int32),
    ])
    def eval_step(inp_colors, inp_idxs, tar_idxs):
        inp_seq_len = tf.shape(inp_idxs)[-1]
        tar_seq_len = tf.shape(tar_idxs)[-1]
        enc_a_mask = models.get_mask(models.MASK_BACKWARD_EQUAL, inp_seq_len, inp_seq_len)
        dec_mask = models.get_mask(models.MASK_NONE, inp_seq_len, tar_seq_len)
        return model([inp_colors, inp_idxs, tar_idxs, enc_a_mask, dec_mask], training=False)
    
    @tf.function
    def train_step_distributed():
        inputs = next(ds_train)
        strategy = tf.distribute.get_strategy()
        per_replica_losses = strategy.run(train_step_normal, args=(inputs,))
        return tf.reduce_mean(strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None) / config['num_devices'])
    
    return train_step_grad_accum, train_step_normal, train_step_distributed, eval_step

class TrainingLoop():
    
    def __init__(self, config, viz, model, optimizer, ds_train, ds_test, ds_test_shuffled, batch_size_schedule, model_name):
        
        self.config = config
        self.viz = viz
        self.optimizer = optimizer
        
        self.model_name = model_name
        
        unshuffled_colors, unshuffled_idxs, _ = next(iter(ds_test))
        shuffled_colors, shuffled_idxs, _ = next(iter(ds_test_shuffled))
        
        self.test_colors = tf.concat([unshuffled_colors[:5], shuffled_colors[:5]], axis=0)
        self.test_idxs = tf.concat([unshuffled_idxs[:5], shuffled_idxs[:5]], axis=0)
        
        n_epochs, steps_per_epoch = self.config['n_epochs'], self.config['steps_per_epoch']
        
        self.step_index = 0
        self.last_eval_loss = None
        self.loss_history = np.zeros([n_epochs*steps_per_epoch])
        self.last_eval_step = 0
        self.last_log_index = 0
        self.running_mean = 0
        self.accum_steps = config.start_accum_steps
        
        self.batch_size_schedule = batch_size_schedule
        
        self.wandb_init()
        
        self.train_step_grad_accum, self.train_step_normal, self.train_step_distributed, self.eval_step = training_functions(config, model, optimizer, ds_train)
    
    def wandb_init(self):
         if self.config['use_wandb']:
            wandb_id = wandb.util.generate_id()
            wandb.init(project='dist-mnist', entity='maxeonyx', name=self.model_name + '-' + wandb_id, config=self.config)
    
    def update_infobar(self, info_bar, loss, loss_mean, learning_rate, minibatch_size, accum_steps):
        window_size = self.config['loss_window_size']
        num_replicas = self.config['num_devices']
        # {learning_rate:.6f}
        info_bar.update(f'Loss: {loss:.5f}, Loss ({window_size} step avg.): {loss_mean:.5f}, Learning Rate: ..., #R*BS*GA: {num_replicas:>3}*{minibatch_size}*{accum_steps:<5}')
            
    def evaluate(self, inp_colors, all_idxs, manager=None):
        n = inp_colors.shape[-1]
        n_total = all_idxs.shape[-1]
        out_colors = inp_colors

        if manager is None:
            manager = enlighten.get_manager()
        evaluate_counter = manager.counter(total=n, desc="Evaluating", unit='pixels', leave=False)
        for i in evaluate_counter(range(n, n_total)):

            inp_idxs = all_idxs[:, :i]
            tar_idxs = all_idxs[:, i:]

            logits = self.eval_step(out_colors, inp_idxs, tar_idxs)
            # shape: (batch_size, tar_seq_length, n_colors)
            colors = logits
            # apply softmax on the logits and sample from the distribution
            predictions = tf.random.categorical(logits[:, 0], 1, dtype=tf.int32)
            # append prediction
            out_colors = tf.concat([out_colors, predictions], axis=-1)

        evaluate_counter.close()
        return out_colors

    def process_batch(self, show_input=True, show_output=True, manager=None):
        all_colors = self.test_colors
        all_idxs = self.test_idxs
        batch_size = all_colors.shape[0]
        n = all_colors.shape[-1] // 2 
        half_colors = all_colors[:, :n]
        half_idxs = all_idxs[:, :n]
        image_width, image_height = self.config['image_width'], self.config['image_height']
        if show_input:
            self.viz.showSeq(half_colors, half_idxs, (image_width, image_height), batch_size, unshuffle=True)
            self.viz.showSeq(all_colors, all_idxs, (image_width, image_height), batch_size, unshuffle=True)
        if show_output:
            result_colors = self.evaluate(half_colors, all_idxs, manager=manager)
            return self.viz.showSeq(result_colors, all_idxs, (image_width, image_height), batch_size, unshuffle=True)

    def train(self):
        self.process_batch(show_output=False)
        n_epochs, steps_per_epoch = self.config['n_epochs'], self.config['steps_per_epoch']
        with enlighten.get_manager() as manager:
            status_bar = manager.status_bar(f"Training model '{self.model_name}'", justify=enlighten.Justify.CENTER)
            info_bar = manager.status_bar('Loss: ??????, Learning Rate: ???????, Batch Size: ???*?????')
            steps_bar = manager.counter(total=n_epochs*steps_per_epoch, count=self.step_index, desc='Steps', color='green', unit='steps')
            
            current_epoch = self.step_index // steps_per_epoch
            while current_epoch < n_epochs:
                epoch_bar = manager.counter(total=steps_per_epoch, count=self.step_index%steps_per_epoch, desc=f'Epoch {current_epoch:<3}', color='blue', unit='steps', leave=False)
                while self.step_index < (current_epoch+1)*steps_per_epoch+self.step_index%steps_per_epoch:
                    
                    self._train_inner(info_bar, manager)
                    
                    epoch_bar.update()
                    steps_bar.update()
                    self.step_index += 1
                    
                current_epoch = self.step_index // steps_per_epoch
                epoch_bar.close()
        
    def _train_inner(self, info_bar, manager):
        
        minibatch_size = self.config['minibatch_size']
        window_size = self.config['loss_window_size']
        
        loss=0
        
        if self.config['batch_size_schedule'] is None:
            loss = self.train_step_distributed()
            self.accum_steps = 1
        else:
            if self.config['batch_size_schedule'] == 'dynamic':
                # dynamic batch size
                # increase batch size whenever the 200-step average loss goes up
                if self.step_index > 1 and self.running_mean > self.prev_running_mean and self.accum_steps < self.config['end_accum_steps']:
                    self.accum_steps = max(int(self.accum_steps * 1.5), self.accum_steps+1)
            else:
                self.accum_steps = self.batch_size_schedule(self.step_index)
            self.update_infobar(info_bar, loss, self.running_mean, self.optimizer._decayed_lr(tf.float32), minibatch_size, self.accum_steps)
            loss = self.train_step_grad_accum(self.accum_steps)
        
        self.loss_history[self.step_index] = loss
        if self.step_index > 0:
            self.prev_running_mean = self.running_mean
        self.running_mean = np.mean(self.loss_history[max(0, self.step_index-window_size) : self.step_index+1])

        self.update_infobar(info_bar, loss, self.running_mean, self.optimizer._decayed_lr(tf.float32), minibatch_size, self.accum_steps)
        
        if self.last_eval_loss is None or (self.running_mean < 1 and self.running_mean <= self.last_eval_loss * 0.9 and self.step_index >= self.last_eval_step + 30):
            self.last_eval_loss = self.running_mean
            self.last_eval_step = self.step_index
            print(f"Step {self.step_index}, Loss ({window_size} step avg.): {self.last_eval_loss}")
            fig = self.process_batch(show_input=False, manager=manager)
            if self.config['use_wandb']:
                self.last_log_index = self.step_index
                wandb.log({'loss': self.running_mean, 'learning_rate': self.optimizer._decayed_lr(tf.float32), 'image_eval': wandb.Image(fig)}, step=self.step_index)
                    
        if self.config['use_wandb'] and self.step_index > self.last_log_index + self.config['wandb_log_interval']:
            self.last_log_index = self.step_index
            wandb.log({'loss': self.running_mean, 'learning_rate': self.optimizer._decayed_lr(tf.float32)}, step=self.step_index)
