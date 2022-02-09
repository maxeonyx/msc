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

import schedules
import models

def log2(x):
    numerator = tf.math.log(x)
    denominator = tf.math.log(tf.constant(2, dtype=numerator.dtype))
    return numerator / denominator

def entropy_of_probabilities(probabilities, do_print=False):
    if do_print:
        print("p(x) =", probabilities)
    log_probabilities = log2(probabilities)
    if do_print:
        print("log p(x) =", log_probabilities)
    entropies = -1 * tf.reduce_sum(probabilities*log_probabilities, axis=-1)
    entropies = tf.expand_dims(entropies, axis=-1)
    
    entropies = entropies / log2(tf.cast(probabilities.shape[-1], tf.float32))
    
    if do_print:
        print("H(X) =", entropies)
    return entropies

def entropy_of_logits(logits, do_print=False):
    if do_print:
        print("logits =", logits)
    
    probabilities = tf.nn.softmax(logits, axis=-1)
    
    return entropy_of_probabilities(probabilities, do_print)

def wandb_init(config, model_name, resume):
    if config['use_wandb']:
        wandb.init(project='dist-mnist', entity='maxeonyx', name=model_name, config=config, resume=resume)
        
class Evaluator():
        
    def new_test_batch(self):
        unshuffled_colors, unshuffled_idxs, shuffled_colors, shuffled_idxs, _ = next(self.ds_test)
        
        self.test_colors = tf.concat([unshuffled_colors[:5], shuffled_colors[:5]], axis=0)
        self.test_idxs = tf.concat([unshuffled_idxs[:5], shuffled_idxs[:5]], axis=0)
    
    def __init__(self, config, model, optimizer, viz, ds, ds_train, ds_test):
        
        self.config = config
        self.model = model
        self.optimizer = optimizer
        ds_train = iter(ds_train)
        self.viz = viz
        self.ds = ds
        self.ds_train = ds_train
        ds_test = iter(ds_test)
        self.ds_test = ds_test
        
        dtype = tf.float32# if config.dtype == 'float32' else tf.float16
        
        self.new_test_batch()

        loss_function = keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
        weights = model.trainable_weights
    
        @tf.function
        def train_step_inner_inner(x_inp, x_tar, i_inp, i_tar, enc_mask_type, dec_mask_type):
            inp_seq_len = tf.shape(i_inp)[-1]
            tar_seq_len = tf.shape(i_tar)[-1]
            enc_mask = models.get_mask(enc_mask_type, inp_seq_len, inp_seq_len)
            dec_mask = models.get_mask(dec_mask_type, inp_seq_len, tar_seq_len)
            with tf.GradientTape() as tape:
                x_out = model([x_inp, i_inp, i_tar, enc_mask, dec_mask], training=True)
                loss = loss_function(x_tar, x_out)
                scaled_loss = optimizer.get_scaled_loss(loss)
                gradients = tape.gradient(scaled_loss, weights)
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

            steps = tf.constant(0)
            accum_gradients = [tf.zeros_like(w) for w in weights]
            accum_loss = tf.constant(0, dtype)

            if config.training_mode == 'query_next' or config.training_mode == 'full_combination':
                steps += 1
                enc_mask_type = models.MASK_BACKWARD_EQUAL
                dec_mask_type = models.MASK_BACKWARD
                x_inp = colors_inp[:, :]
                x_tar = colors_tar[:, :]
                i_inp = idxs[:, :]
                i_tar = idxs[:, :]

                loss, gradients = train_step_inner_inner(x_inp, x_tar, i_inp, i_tar, enc_mask_type, dec_mask_type)
                accum_gradients = [accum_grad+grad for accum_grad, grad in zip(accum_gradients, gradients)]
                accum_loss += tf.reduce_mean(loss)

            if config.training_mode == 'query_unknown':
                steps += 1
                enc_mask_type = models.MASK_NONE
                dec_mask_type = models.MASK_NONE
                x_inp = colors_inp[:, :n]
                x_tar = colors_tar[:, n:]
                i_inp = idxs[:, :n]
                i_tar = idxs[:, n:]
                loss, gradients = train_step_inner_inner(x_inp, x_tar, i_inp, i_tar, enc_mask_type, dec_mask_type)
                accum_gradients = [accum_grad+grad for accum_grad, grad in zip(accum_gradients, gradients)]
                accum_loss += tf.reduce_mean(loss)

            if config.training_mode == 'query_all' or config.training_mode == 'full_combination':
                steps += 1
                enc_mask_type = models.MASK_NONE
                dec_mask_type = models.MASK_NONE
                x_inp = colors_inp[:, :n]
                x_tar = colors_tar[:, :]
                i_inp = idxs[:, :n]
                i_tar = idxs[:, :]
                loss, gradients = train_step_inner_inner(x_inp, x_tar, i_inp, i_tar, enc_mask_type, dec_mask_type)
                accum_gradients = [accum_grad+grad for accum_grad, grad in zip(accum_gradients, gradients)]
                accum_loss += tf.reduce_mean(loss)
            
            float_steps = tf.cast(steps, dtype)
            # without dividing, learning rate implicitly changes if batch size changes
            accum_gradients = [accum_grad / float_steps for accum_grad in accum_gradients]
            accum_loss /= float_steps
            return accum_loss, accum_gradients


        @tf.function
        def train_step_grad_accum(batch, accum_steps):
            float_steps = tf.cast(accum_steps, dtype)
            accum_gradients = [tf.zeros_like(w) for w in weights]
            accum_loss = tf.constant(0, dtype)
            for step in tf.range(accum_steps):
                colors, idxs, colors_shuf, idxs_shuf, shuffled_colors_noise, n = batch
                inputs = colors[step], idxs[step], colors_shuf[step], idxs_shuf[step], shuffled_colors_noise[step], n[step]
                loss, gradients = train_step_combination(inputs)
                accum_gradients = [accum_grad+grad for accum_grad, grad in zip(accum_gradients, gradients)]
                accum_loss += tf.reduce_mean(loss)
            # without dividing, learning rate implicitly changes if batch size changes
            accum_gradients = [accum_grad / float_steps for accum_grad in accum_gradients]
            gradients = optimizer.get_unscaled_gradients(accum_gradients)
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
            enc_a_mask = models.get_mask(models.MASK_NONE, inp_seq_len, inp_seq_len)
            dec_mask = models.get_mask(models.MASK_NONE, inp_seq_len, tar_seq_len)
            return model([inp_colors, inp_idxs, tar_idxs, enc_a_mask, dec_mask], training=False)

        @tf.function
        def train_step_distributed():
            inputs = next(ds_train)
            strategy = tf.distribute.get_strategy()
            per_replica_losses = strategy.run(train_step_normal, args=(inputs,))
            return tf.reduce_mean(strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None) / float(config.num_devices))

        @tf.function
        def train_step_distributed_accum(accum_steps):
            inputs = next(ds_train)
            strategy = tf.distribute.get_strategy()
            per_replica_losses = strategy.run(train_step_grad_accum, args=(inputs, accum_steps))
            return tf.reduce_mean(strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None) / (float(config.num_devices) * float(accum_steps)))
        
        self.train_step_normal = train_step_normal
        self.train_step_grad_accum = train_step_grad_accum
        self.train_step_distributed = train_step_distributed
        self.train_step_distributed_accum = train_step_distributed_accum
        self.eval_step = eval_step
    
    
    
    def test_loss(self, manager=None):
        """
        Calculate the average loss across a variety evaluation modes on the test data.
        - Order eg. shuffled & sequential
        - # of Conditioning pixels e.g. [1, 8, 16, 32]
        """
        
        n_batches = self.config.test_size // self.config.test_minibatch_size
        total_steps = 0
        for shuffled, n_seq in [(True, self.config.test_n_shuf), (False, self.config.test_n_seq)]:
            if self.config.test_autoregressive:
                # autoregressive completion takes 784-n steps for
                total_steps += sum(n_batches * self.config.dataset.seq_length - n for n in n_seq)
            else:
                total_steps += n_batches * len(n_seq)
        
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
                    for pix_i in range(self.config.dataset.seq_length - n):
                        pred_logits = self.eval_step(inp_colors, inp_idxs, tar_idxs)

                        if pix_i == 0:
                            loss = loss_function(tar_colors, pred_logits)
                            name = f"loss_{typ}_{n}"
                            if batch_i == 0:
                                losses[name] = loss
                            else:
                                losses[name] = tf.concat([losses[name], loss], axis=0)
                        
                        evaluate_counter.update()
                        if not self.config.test_autoregressive:
                            break
                        
                        this_samples = tf.random.categorical(pred_logits[:, 0], 1, dtype=tf.int32)
                        this_samples = tf.one_hot(this_samples, depth=self.config.dataset.n_colors)
                        if samples is None:
                            samples = this_samples
                        else:
                            samples = tf.concat([samples, this_samples], axis=1)
                        
                    
                    if not self.config.test_autoregressive:
                        continue
                    
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
        """Autoregressively sample a batch of images, each seeded with `inp_colors`"""
        n = inp_colors.shape[-1]
        n_total = all_idxs.shape[-1]
        autoregressive_samples = inp_colors

        if manager is None:
            manager = enlighten.get_manager()
        evaluate_counter = manager.counter(total=n_total-n+1, desc="Evaluating", unit='steps', leave=False)
        
        # compute the expected color across all unknown pixels
        inp_idxs = all_idxs[:, :n]
        tar_idxs = all_idxs[:, n:] # target is all unknown pixels
        logits = self.eval_step(autoregressive_samples, inp_idxs, tar_idxs)
        probabilities = tf.nn.softmax(logits, axis=2)

        expected_col = self.ds.expected_col(probabilities)
        expected_col = tf.concat([self.ds.unquantize(inp_colors, to="grayscale"), expected_col], axis=1)
        
        evaluate_counter.update()
        
        for i in range(n, n_total):

            inp_idxs = all_idxs[:, :i]
            tar_idxs = all_idxs[:, i:i+1] # target is first unknown pixel only

            logits = self.eval_step(autoregressive_samples, inp_idxs, tar_idxs)
            
            # applies softmax on the logits and sample from the distribution
            samples = tf.random.categorical(logits[:, 0], 1, dtype=tf.int32)
            autoregressive_samples = tf.concat([autoregressive_samples, samples], axis=-1)
            
            evaluate_counter.update()
            
        evaluate_counter.close()
        return autoregressive_samples, expected_col
    
    def evaluate_varying(self, all_colors, all_idxs, n_fn, manager=None):
        """Autoregressively sample a bunch of images, each starting with n
           (produced by n_fn) real pixels"""
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

            logits = self.eval_step(inp_colors, inp_idxs, tar_idxs)
            unq_inp = self.ds.unquantize(inp_colors)
            unq_inp_rgb = self.ds.to_grayscale_rgb(unq_inp)
            bg_input = self.viz.scatter_on_bg(unq_inp_rgb, inp_idxs, output_length=seq_length)
            if bg_all is None:
                bg_all = bg_input
            else:
                bg_all = tf.concat([bg_all, bg_input], axis=0)

            # softmax along color dimension
            probabilities = tf.nn.softmax(logits, axis=-1)
            expected_col = self.ds.expected_col(probabilities)
                
            expected_col = tf.concat([unq_inp, expected_col], axis=-2)
            
            all_expected_col = expected_col if all_expected_col is None else tf.concat([all_expected_col, expected_col], axis=0)

        evaluate_counter.close()
        return all_expected_col, bg_all
    
    def evaluate_varying_entropy(self, all_colors, all_idxs, n_fn, manager=None):
        """Sample a bunch of images, each starting with n
           (produced by n_fn) real pixels. Show the entropy over all outputs, including the inputs."""
        batch_size = all_colors.shape[0]
        seq_length = all_colors.shape[1]
        
        if manager is None:
            manager = enlighten.get_manager()
        evaluate_counter = manager.counter(total=batch_size, desc="Evaluating", unit='steps', leave=False)
        
        all_entropies = None
        for i in evaluate_counter(range(batch_size)):
            n = min(max(1, int(n_fn(i))), all_colors.shape[1] - 1)
            inp_idxs = all_idxs[:1, :n]
            inp_colors = all_colors[:1, :n]
            
            tar_idxs = all_idxs[:1, :]

            logits = self.eval_step(inp_colors, inp_idxs, tar_idxs)
            
            entropies = entropy_of_logits(logits)
                
            all_entropies = entropies if all_entropies is None else tf.concat([all_entropies, entropies], axis=0)

        evaluate_counter.close()
        return all_entropies

    def process_batch(self, return_figs=False, show_input=True, show_output=True, manager=None):
        all_colors = self.test_colors
        all_idxs = self.test_idxs
        batch_size = all_colors.shape[0]
        n = all_colors.shape[-1] // 2
        
        half_colors = all_colors[:, :n]
        half_idxs = all_idxs[:, :n]
        image_height, image_width = self.config.dataset.image_size
        figs = []
        if show_input:
            self.viz.showSeq(half_colors, half_idxs, (image_width, image_height), batch_size, unshuffle=True)
            self.viz.showSeq(all_colors, all_idxs, (image_width, image_height), batch_size, unshuffle=True)
        if show_output:
            autoregressive_samples, expected_col = self.evaluate(half_colors, all_idxs, manager=manager)
            figs.append(self.viz.showSeq(autoregressive_samples, all_idxs, (image_width, image_height), batch_size, unshuffle=True, return_fig=True))
            figs.append(self.viz.showSeq(expected_col,           all_idxs, (image_width, image_height), batch_size, unshuffle=True, do_unquantize=False, return_fig=True))
            repeated_col = tf.tile(all_colors[5:6], [batch_size, 1])
            repeated_idxs = tf.tile(all_idxs[5:6], [batch_size, 1])
            varying_n, varying_in = self.evaluate_varying(repeated_col, repeated_idxs, n_fn=lambda i: self.config.dataset.seq_length//(batch_size+1)*(i+1), manager=manager)
            e_varying_n = self.evaluate_varying_entropy(repeated_col, repeated_idxs, n_fn=lambda i: self.config.dataset.seq_length//(batch_size+1)*(i+1), manager=manager)
            if show_input:
                self.viz.showSeq(varying_in,                repeated_idxs, (image_width, image_height), batch_size, unshuffle=False, do_unquantize=False)
            figs.append(self.viz.showSeq(varying_n,              repeated_idxs, (image_width, image_height), batch_size, unshuffle=True, do_unquantize=False, return_fig=True))
            figs.append(self.viz.showSeq(e_varying_n,              repeated_idxs, (image_width, image_height), batch_size, unshuffle=True, do_unquantize=False, return_fig=True))
            repeated_col = tf.tile(all_colors[0:1], [batch_size, 1])
            repeated_idxs = tf.tile(all_idxs[0:1], [batch_size, 1])
            varying_n, varying_in = self.evaluate_varying(repeated_col, repeated_idxs, n_fn=lambda i: self.config.dataset.seq_length//(batch_size+1)*(i+1), manager=manager)
            e_varying_n = self.evaluate_varying_entropy(repeated_col, repeated_idxs, n_fn=lambda i: self.config.dataset.seq_length//(batch_size+1)*(i+1), manager=manager)
            if show_input:
                self.viz.showSeq(varying_in,                repeated_idxs, (image_width, image_height), batch_size, unshuffle=False, do_unquantize=False)
            figs.append(self.viz.showSeq(varying_n,              repeated_idxs, (image_width, image_height), batch_size, unshuffle=True, do_unquantize=False, return_fig=True))
            figs.append(self.viz.showSeq(e_varying_n,              repeated_idxs, (image_width, image_height), batch_size, unshuffle=True, do_unquantize=False, return_fig=True))
            if return_figs:
                return figs
    
    def many_autoregressive(self, n_seq=None):
        all_colors = self.test_colors
        all_idxs = self.test_idxs
        batch_size = all_colors.shape[0]
        image_height, image_width = self.config.dataset.image_size
        if n_seq is None:
            n_seq = [15*i + 1 for i in range(5,10)]
        for n in n_seq:
            inp_colors = all_colors[:, :n]
            inp_idxs = all_idxs[:, :n]
            autoregressive_samples, _ = self.evaluate(inp_colors, all_idxs)
            print("n =", n)
            self.viz.showSeq(inp_colors, inp_idxs, (image_width, image_height), batch_size, unshuffle=True)
            self.viz.showSeq(autoregressive_samples, all_idxs, (image_width, image_height), batch_size, unshuffle=True)

class TrainingLoop():
    
    def __init__(self, config, evaler, model_name):
        
        self.config = config
        self.evaler = evaler
        
        self.model_name = model_name
        
        wandb_init(config, model_name, resume=False)
        
        self.step_index = 0
        self.last_eval_loss = None
        self.loss_history = np.zeros([config.n_steps])
        self.last_eval_step = 0
        self.last_log_index = 0
        self.accum_steps = config.start_accum_steps
        self.running_mean = 0.
        self.test_losses = {}
    
    def update_infobar(self, info_bar, loss):
        
        def format_loss(l):
            l_str = f"{l:.5g}"
            l_format = f"{l_str + ',': <8}"
            return l_format
        
        info_bar_text = ""
        
        loss_format = format_loss(loss)
        info_bar_text += f"Loss: {loss_format} "
        
        window_size = self.config['loss_window_size']
        running_mean_format = format_loss(self.running_mean)
        info_bar_text += f"Loss (mean) ({window_size} steps): {running_mean_format} "
        
        log_loss_format = format_loss(tf.math.log(self.running_mean))
        info_bar_text += f"Log Loss (mean): {log_loss_format: <9} "
        
        learning_rate_format = format_loss(self.learning_rate)
        info_bar_text += f"Learning Rate: {learning_rate_format: <9} "
        
        shuf_key = f'loss_shuf_{self.config.test_n_shuf[0]}'
        if shuf_key in self.test_losses:
            test_loss_shuf = format_loss(self.test_losses[shuf_key])
            info_bar_text += f"Test Loss (shuf): {test_loss_shuf} "
        
        seq_key = f'loss_seq_{self.config.test_n_seq[0]}'
        if seq_key in self.test_losses:
            test_loss_seq = format_loss(self.test_losses[seq_key])
            info_bar_text += f"Test Loss (seq): {test_loss_seq} "
        
        num_replicas = self.config['num_devices']
        minibatch_size = self.config.minibatch_size
        accum_steps = self.accum_steps
        batch_size_text = f"{num_replicas}*{minibatch_size}*{accum_steps}"
        info_bar_text += f"Batch Size: {batch_size_text} (Devices*Minibatch*GradAccum)"
        
        info_bar.update(info_bar_text)
        
        
    def train(self):
        
        with enlighten.get_manager() as manager:
            status_bar = manager.status_bar(f"Training model '{self.model_name}'", justify=enlighten.Justify.CENTER)
            info_bar = manager.status_bar('Loss: ??????, Learning Rate: ???????, Batch Size: ???*?????')
            steps_bar = manager.counter(total=self.config.n_steps, count=self.step_index, desc='Steps', color='green', unit='steps')
            
            while self.step_index < self.config.n_steps:
                
                self.learning_rate = self.evaler.optimizer.inner_optimizer._decayed_lr(tf.float32)
                self._train_inner(info_bar, manager)

                steps_bar.update()
                self.step_index += 1
        
    def _train_inner(self, info_bar, manager):
        
        minibatch_size = self.config['minibatch_size']
        window_size = self.config['loss_window_size']
        
        loss=0.
        
        if self.config.grad_accum_steps is None or self.config.grad_accum_steps == 1:
            loss = self.evaler.train_step_distributed()
            self.accum_steps = 1
        else:
            if type(self.config.grad_accum_steps) is int:
                self.accum_steps = self.config.grad_accum_steps
            elif self.config.grad_accum_steps == 'dynamic':
                # dynamic batch size
                # increase batch size whenever the 200-step average loss goes up
                # only start doing it after the 20th step, because the training starts of very unstable
                if self.step_index > 20 and self.running_mean > self.prev_running_mean and self.accum_steps < self.config.max_accum_steps:
                    self.accum_steps = min(max(int(self.accum_steps * 1.5), self.accum_steps+1), self.config.max_accum_steps)
            else:
                schedule_name, *params = self.config.grad_accum_steps
                bs_sched = schedules.batch_size_schedule(self.config, schedule_name, params)
                self.accum_steps = bs_sched(self.step_index)
            self.update_infobar(info_bar, loss)
            loss = self.evaler.train_step_distributed_accum(self.accum_steps)
        
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
            fig1, fig2, fig3, fig4, fig5, fig6 = self.evaler.process_batch(return_figs=True, show_input=(self.step_index == 0), manager=manager)
            if self.config['use_wandb']:
                self.last_log_index = self.step_index
                wandb.log({
                    'log_loss': tf.math.log(loss),
                    'learning_rate': self.learning_rate,
                    'viz_autoregressive': wandb.Image(fig1),
                    'viz_expected': wandb.Image(fig2),
                    'viz_varying': wandb.Image(fig3),
                    'viz_varying_entropy': wandb.Image(fig4),
                    'viz_varying_sequential': wandb.Image(fig5),
                    'viz_varying_sequential_entropy': wandb.Image(fig6),
                }, step=self.step_index)
                    
        if self.config['use_wandb'] and self.step_index > 0 and self.step_index > self.last_log_index + self.config['wandb_log_interval']:
            self.last_log_index = self.step_index
            wandb_interval_mean_loss = np.mean(self.loss_history[max(0, self.step_index-self.config['wandb_log_interval']) : self.step_index+1])
            wandb.log({'log_loss_mean': tf.math.log(wandb_interval_mean_loss), 'loss_mean': wandb_interval_mean_loss, 'loss': loss, 'learning_rate': self.learning_rate}, step=self.step_index)
                    
        if self.step_index > 0 and self.step_index % self.config.test_interval == 0:
            losses = self.evaler.test_loss(manager=manager)
            self.test_losses = losses
            
            wandb.log(losses, step=self.step_index)
