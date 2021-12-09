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

def wandb_init(config, model_name, resume):
    if config['use_wandb']:
        wandb.init(project='dist-mnist', entity='maxeonyx', name=model_name, config=config, resume=resume)

def training_functions(config, model, optimizer, ds_train):
    
    loss_function = keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
    weights = model.trainable_weights
    
    ds_train_iter = iter(ds_train)
    
    @tf.function
    def train_step_inner_inner(x_inp, x_tar, i_inp, i_tar, enc_mask_type, dec_mask_type):
        inp_seq_len = tf.shape(i_inp)[-1]
        tar_seq_len = tf.shape(i_tar)[-1]
        enc_mask = models.get_mask(enc_mask_type, inp_seq_len, inp_seq_len)
        dec_mask = models.get_mask(dec_mask_type, inp_seq_len, tar_seq_len)
        with tf.GradientTape() as tape:
            x_out = model([x_inp, i_inp, i_tar, enc_mask, dec_mask], training=True)
            loss = loss_function(x_tar, x_out)
            gradients = tape.gradient(loss, weights)
        return loss, gradients
        
    @tf.function
    def train_step_combination(inputs):
        colors, idxs, colors_shuf, idxs_shuf, shuffled_colors_noise, n = inputs
        idxs = idxs_shuf
        colors_tar = colors_shuf
        colors_inp = colors_shuf
        if config.add_noise:
            colors_inp = shuffled_colors_noise
        n = tf.squeeze(n)
        
        float_steps = tf.constant(0, tf.float32)
        accum_gradients = [tf.zeros_like(w, dtype=tf.float32) for w in weights]
        accum_loss = tf.constant(0, tf.float32)
        
        if config.training_mode == 'query_next' or config.training_mode == 'combination':
            float_steps += 1.0
            enc_mask_type = models.MASK_BACKWARD_EQUAL
            dec_mask_type = models.MASK_BACKWARD
            x_inp = colors_inp[:, :]
            x_tar = colors_tar[:, :]
            i_inp = idxs[:, :]
            i_tar = idxs[:, :]
            
#             if config.add_noise:
#                 tf.shape(x_inp)
#                 amount_of_noise = int(config.noise_fraction * tf.shape(x_inp)[1])
                
#                 idxs = tf.random.shuffle(i_inp)[:amount_of_noise]
#                 random_colors = tf.random.uniform(
#                 x_inp = tf.tensor_scatter_nd_update(x_inp, noise, 
#                 # tf.tesnor_scatter_nd_update
            loss, gradients = train_step_inner_inner(x_inp, x_tar, i_inp, i_tar, enc_mask_type, dec_mask_type)
            accum_gradients = [accum_grad+grad for accum_grad, grad in zip(accum_gradients, gradients)]
            accum_loss += tf.reduce_mean(loss)
        
        if config.training_mode == 'query_all' or config.training_mode == 'combination':
            float_steps += 1.0
            enc_mask_type = models.MASK_NONE
            dec_mask_type = models.MASK_NONE
            x_inp = colors_inp[:, :n]
            x_tar = colors_tar[:, n:]
            i_inp = idxs[:, :n]
            i_tar = idxs[:, n:]
            loss, gradients = train_step_inner_inner(x_inp, x_tar, i_inp, i_tar, enc_mask_type, dec_mask_type)
            accum_gradients = [accum_grad+grad for accum_grad, grad in zip(accum_gradients, gradients)]
            accum_loss += tf.reduce_mean(loss)
            
        # uncomment to divide gradient.
        # without dividing, learning rate implicitly changes if batch size changes
        accum_gradients = [accum_grad / float_steps for accum_grad in accum_gradients]
        accum_loss /= float_steps
        return accum_loss, accum_gradients
    
    @tf.function
    def train_step_query_all(inputs):
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
    def train_step_grad_accum(batch, minibatch_size, accum_steps):
        float_steps = tf.cast(accum_steps, tf.float32)
        accum_gradients = [tf.zeros_like(w, dtype=tf.float32) for w in weights]
        accum_loss = tf.constant(0, tf.float32)
        for step in tf.range(accum_steps):
            colors, idxs, n = batch
            inputs = colors[step], idxs[step], n[step]
            loss, gradients = train_step_inner(inputs)
            accum_gradients = [accum_grad+grad for accum_grad, grad in zip(accum_gradients, gradients)]
            accum_loss += tf.reduce_mean(loss)
        # uncomment to divide gradient. without dividing, learning rate implicitly changes
        accum_gradients = [accum_grad / float_steps for accum_grad in accum_gradients]
        optimizer.apply_gradients(zip(accum_gradients, weights))
        return accum_loss
    
    @tf.function
    def train_step_normal(inputs):
        loss, gradients = train_step_combination(inputs)
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
        inputs = next(ds_train_iter)
        strategy = tf.distribute.get_strategy()
        per_replica_losses = strategy.run(train_step_normal, args=(inputs,))
        return tf.reduce_mean(strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None) / config['num_devices'])
    
    @tf.function
    def train_step_distributed_accum(accum_steps):
        inputs = next(ds_train_iter)
        strategy = tf.distribute.get_strategy()
        per_replica_losses = strategy.run(train_step_grad_accum, args=(inputs, config['minibatch_size'], accum_steps))
        return tf.reduce_mean(strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None) / (config['num_devices'] * accum_steps))
    
    return train_step_grad_accum, train_step_normal, train_step_distributed, train_step_distributed_accum, eval_step

class TrainingLoop():
    
    def __init__(self, config, viz, model, optimizer, ds, ds_train, ds_test, batch_size_schedule, model_name):
        
        self.config = config
        self.viz = viz
        self.ds = ds
        self.optimizer = optimizer
        
        self.model_name = model_name
        
        self.ds_test = iter(ds_test)
        
        self.new_test_batch()
        
        self.step_index = 0
        self.last_eval_loss = None
        self.loss_history = np.zeros([config.n_steps])
        self.last_eval_step = 0
        self.last_log_index = 0
        self.accum_steps = config.start_accum_steps
        self.running_mean = 0.
        self.test_loss_shuf = 0.
        self.test_loss_seq = 0.
        
        self.batch_size_schedule = batch_size_schedule
        
        self.train_step_grad_accum, self.train_step_normal, self.train_step_distributed, self.train_step_distributed_accum, self.eval_step = training_functions(config, model, optimizer, ds_train)
        
    def new_test_batch(self):
        unshuffled_colors, unshuffled_idxs, shuffled_colors, shuffled_idxs, _ = next(self.ds_test)
        
        self.test_colors = tf.concat([unshuffled_colors[:5], shuffled_colors[:5]], axis=0)
        self.test_idxs = tf.concat([unshuffled_idxs[:5], shuffled_idxs[:5]], axis=0)
    
    def update_infobar(self, info_bar, loss):
        window_size = self.config['loss_window_size']
        num_replicas = self.config['num_devices']
        minibatch_size = self.config.minibatch_size
        loss = tf.math.log(loss)
        running_mean = tf.math.log(self.running_mean)
        test_loss_shuf = tf.math.log(self.test_loss_shuf)
        test_loss_seq = tf.math.log(self.test_loss_seq)
        
        info_bar.update(f'Loss: {loss:.5g}, Loss ({window_size} step avg.): {running_mean:.5g}, Test Loss (shuf): {test_loss_shuf:.5g}, Test Loss (seq): {test_loss_seq:.5g}, #R*BS*GA: {num_replicas:>3}*{minibatch_size}*{self.accum_steps:<5}')
        
    def test_loss(self, manager=None):
        
        n_batches = self.config.test_size // self.config.test_minibatch_size
        total_steps = 0
        for shuffled, n_seq in [(True, self.config.test_n_shuf), (False, self.config.test_n_seq)]:
            # autoregressive completion takes 784-n steps for 
            total_steps += sum(n_batches * self.config.seq_length - n for n in n_seq)
        
        if manager is None:
            manager = enlighten.get_manager()
        evaluate_counter = manager.counter(total=total_steps, desc="Evaluating", unit='steps', leave=False)
        
        loss_function = keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
        
        losses = {}
        for shuffled, typ, n_seq in [(True, "shuf", self.config.test_n_shuf), (False, "seq", self.config.test_n_seq)]:
            for i, n in enumerate(n_seq):
                for batch_i in range(n_batches):
                    colors, idxs, colors_shuf, idxs_shuf, _ = next(self.ds_test)

                    if shuffled:
                        colors, idxs = colors_shuf, idxs_shuf

                    inp_colors = colors[:, :n]
                    tar_colors = colors[:, n:]
                    inp_idxs = idxs[:, :n]
                    tar_idxs = idxs[:, n:]
                    
                    samples = None
                    for pix_i in range(self.config.seq_length - n):
                        pred_logits = self.eval_step(inp_colors, inp_idxs, tar_idxs)

                        if pix_i == 0:
                            loss = loss_function(tar_colors, pred_logits)
                            name = f"loss_{typ}_{n}"
                            if batch_i == 0:
                                losses[name] = loss
                            else:
                                losses[name] = tf.concat([losses[name], loss], axis=0)
                        
                        this_samples = tf.random.categorical(pred_logits[:, 0], 1, dtype=tf.int32)
                        this_samples = tf.one_hot(this_samples, depth=self.config.n_colors)
                        if samples is None:
                            samples = this_samples
                        else:
                            samples = tf.concat([samples, this_samples], axis=1)
                        
                        evaluate_counter.update()
                    
                    loss = loss_function(tar_colors, samples)
                    name = f"loss_autoreg_{typ}_{n}"
                    if batch_i == 0:
                        losses[name] = loss
                    else:
                        losses[name] = tf.concat([losses[name], loss], axis=0)
        
        evaluate_counter.close()
        
        for name in losses:
            losses[name] = tf.reduce_mean(losses[name])
        
        return losses
            
    def evaluate(self, inp_colors, all_idxs, manager=None):
        n = inp_colors.shape[-1]
        n_total = all_idxs.shape[-1]
        autoregressive_samples = inp_colors

        if manager is None:
            manager = enlighten.get_manager()
        evaluate_counter = manager.counter(total=n_total-n, desc="Evaluating", unit='steps', leave=False)
        
        for i in evaluate_counter(range(n, n_total)):

            inp_idxs = all_idxs[:, :i]
            tar_idxs = all_idxs[:, i:]

            logits = self.eval_step(autoregressive_samples, inp_idxs, tar_idxs)
            
            # first time through
            if i == n:
                # softmax along color dimension
                probabilities = tf.nn.softmax(logits, axis=2)
            
            # shape: (batch_size, tar_seq_length, n_colors)
            colors = logits
            
            # apply softmax on the logits and sample from the distribution
            samples = tf.random.categorical(logits[:, 0], 1, dtype=tf.int32)
            autoregressive_samples = tf.concat([autoregressive_samples, samples], axis=-1)

        expected_col = self.ds.expected_col(probabilities)
        expected_col = tf.concat([self.ds.unquantize(inp_colors), expected_col], axis=1)
        expected_col = tf.squeeze(expected_col)
            
        evaluate_counter.close()
        return autoregressive_samples, expected_col
    
    def evaluate_varying(self, all_colors, all_idxs, n_fn, manager=None):
        batch_size = all_colors.shape[0]
        seq_length = all_colors.shape[1]
        
        if manager is None:
            manager = enlighten.get_manager()
        evaluate_counter = manager.counter(total=batch_size, desc="Evaluating", unit='steps', leave=False)
        
        bg_all = None
        
        all_expected_col = None
        for i in evaluate_counter(range(batch_size)):
            n = min(max(1, int(n_fn(i))), all_colors.shape[1] - 1)
            inp_idxs = all_idxs[:1, :n]
            inp_colors = all_colors[:1, :n]
            tar_idxs = all_idxs[:1, n:]
            
            unq_inp = self.viz.unquantize(inp_colors)
            bg_input = self.viz.scatter_on_bg(unq_inp, inp_idxs, output_length=seq_length)
            if bg_all is None:
                bg_all = bg_input
            else:
                bg_all = tf.concat([bg_all, bg_input], axis=0)

            logits = self.eval_step(inp_colors, inp_idxs, tar_idxs)
            
            # softmax along color dimension
            probabilities = tf.nn.softmax(logits, axis=2)
            
            expected_col = self.ds.expected_col(probabilities)
            expected_col = tf.concat([self.ds.unquantize(inp_colors), expected_col], axis=1)
            expected_col = tf.squeeze(expected_col)
            expected_col = tf.expand_dims(expected_col, 0)
            all_expected_col = expected_col if all_expected_col is None else tf.concat([all_expected_col, expected_col], axis=0)

        evaluate_counter.close()
        return all_expected_col, bg_all

    def process_batch(self, return_figs=False, show_input=True, show_output=True, manager=None):
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
            autoregressive_samples, expected_col = self.evaluate(half_colors, all_idxs, manager=manager)
            fig1 = self.viz.showSeq(autoregressive_samples, all_idxs, (image_width, image_height), batch_size, unshuffle=True)
            fig2 = self.viz.showSeq(expected_col,           all_idxs, (image_width, image_height), batch_size, unshuffle=True, do_unquantize=False)
            repeated_col = tf.tile(all_colors[5:6], [batch_size, 1])
            repeated_idxs = tf.tile(all_idxs[5:6], [batch_size, 1])
            varying_n, varying_in = self.evaluate_varying(repeated_col, repeated_idxs, n_fn=lambda i: 2**i + 10, manager=manager)
            if show_input:
                self.viz.showSeq(varying_in,                repeated_idxs, (image_width, image_height), batch_size, unshuffle=False, do_unquantize=False)
            fig3 = self.viz.showSeq(varying_n,              repeated_idxs, (image_width, image_height), batch_size, unshuffle=True, do_unquantize=False)
            repeated_col = tf.tile(all_colors[0:1], [batch_size, 1])
            repeated_idxs = tf.tile(all_idxs[0:1], [batch_size, 1])
            varying_n, varying_in = self.evaluate_varying(repeated_col, repeated_idxs, n_fn=lambda i: 70*i, manager=manager)
            if show_input:
                self.viz.showSeq(varying_in,                repeated_idxs, (image_width, image_height), batch_size, unshuffle=False, do_unquantize=False)
            fig4 = self.viz.showSeq(varying_n,              repeated_idxs, (image_width, image_height), batch_size, unshuffle=True, do_unquantize=False)
            if return_figs:
                return fig1, fig2, fig3, fig4
    
    def many_autoregressive(self, n_seq=None):
        all_colors = self.test_colors
        all_idxs = self.test_idxs
        batch_size = all_colors.shape[0]
        image_width, image_height = self.config['image_width'], self.config['image_height']
        if n_seq is None:
            n_seq = [15*i + 1 for i in range(5,10)]
        for n in n_seq:
            inp_colors = all_colors[:, :n]
            inp_idxs = all_idxs[:, :n]
            autoregressive_samples, _ = self.evaluate(inp_colors, all_idxs)
            print("n =", n)
            self.viz.showSeq(inp_colors, inp_idxs, (image_width, image_height), batch_size, unshuffle=True)
            self.viz.showSeq(autoregressive_samples, all_idxs, (image_width, image_height), batch_size, unshuffle=True)
        

    def train(self):
        self.process_batch(show_output=False)
        with enlighten.get_manager() as manager:
            status_bar = manager.status_bar(f"Training model '{self.model_name}'", justify=enlighten.Justify.CENTER)
            info_bar = manager.status_bar('Loss: ??????, Learning Rate: ???????, Batch Size: ???*?????')
            steps_bar = manager.counter(total=self.config.n_steps, count=self.step_index, desc='Steps', color='green', unit='steps')
            
            while self.step_index < self.config.n_steps:
                    
                self._train_inner(info_bar, manager)

                steps_bar.update()
                self.step_index += 1
        
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
                # only start doing it after the 20th step, because the training starts of very unstable
                if self.step_index > 20 and self.running_mean > self.prev_running_mean and self.accum_steps < self.config['end_accum_steps']:
                    self.accum_steps = min(max(int(self.accum_steps * 1.5), self.accum_steps+1), self.config.end_accum_steps)
            else:
                self.accum_steps = self.batch_size_schedule(self.step_index)
            self.update_infobar(info_bar, loss)
            loss = self.train_step_distributed_accum(self.accum_steps)
        
        self.loss_history[self.step_index] = loss
        if self.step_index > 0:
            self.prev_running_mean = self.running_mean
        self.running_mean = np.mean(self.loss_history[max(0, self.step_index-window_size) : self.step_index+1])

        self.update_infobar(info_bar, loss)
        
        show_images = (
            self.config.display_images
            and (
                self.last_eval_loss is None
                or (
                    self.running_mean < self.config.dont_display_until_loss
                    and self.running_mean <= self.last_eval_loss * 0.9
                    and self.step_index >= self.last_eval_step + 30
                ) or self.step_index >= self.last_eval_step + self.config.display_image_interval
            )
        )
        
        if show_images:
            self.last_eval_loss = self.running_mean
            self.last_eval_step = self.step_index
            print(f"Step {self.step_index}, Loss (last minibatch): {loss}, Loss ({window_size} step avg.): {self.last_eval_loss}")
            fig1, fig2, fig3, fig4 = self.process_batch(return_figs=True, show_input=False, manager=manager)
            if self.config['use_wandb']:
                self.last_log_index = self.step_index
                wandb.log({'log_loss': tf.math.log(loss), 'learning_rate': self.optimizer._decayed_lr(tf.float32), 'viz_autoregressive': wandb.Image(fig1), 'viz_expected': wandb.Image(fig2), 'viz_varying': wandb.Image(fig3), 'viz_varying_sequential': wandb.Image(fig4)}, step=self.step_index)
                    
        if self.config['use_wandb'] and self.step_index > 0 and self.step_index > self.last_log_index + self.config['wandb_log_interval']:
            self.last_log_index = self.step_index
            wandb.log({'log_loss': tf.math.log(loss), 'learning_rate': self.optimizer._decayed_lr(tf.float32)}, step=self.step_index)
                    
        if self.step_index > 0 and self.step_index % self.config.test_interval == 0:
            losses = self.test_loss(manager=manager)
            
            wandb.log(losses, step=self.step_index)
