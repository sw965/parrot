import math
import random
import numpy as np
import tensorflow as tf
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.training import optimizer
from tensorflow.python.ops.clip_ops import clip_by_value
from parrot.dataset import mnist

def random_variable(shape, stddev):
    return tf.Variable(tf.truncated_normal(shape, stddev=stddev))

def zeros_variable(shape):
    return tf.Variable(tf.zeros(shape))

def ones_variable(shape):
    return tf.Variable(tf.ones(shape))

def conv2d(x, filter):
    return tf.nn.conv2d(x, filter, strides=[1, 1, 1, 1], padding="SAME")

def max_pool(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

def batch_normalization(x, training):
    BN = tf.layers.BatchNormalization()
    return BN(x, training=training)

def lrelu(x, alpha=0.01):
    assert 0 <= alpha <= 1.0, "alphaは0~1の範囲でなければならない"
    return tf.maximum(x * alpha, x)

def prelu(x, alpha):
    return tf.maximum(tf.zeros(tf.shape(x)), x) + (tf.minimum(tf.zeros(tf.shape(x)), x) * alpha)

def tanh_exp(x):
    return x * tf.nn.tanh(tf.exp(x))

def flatten(x):
    """
    tf.size(x[0]) の説明 形状がバッチサイズ = 100 縦横奥 5 * 5 * 5 の場合
    1.x[0]で0番目のバッチにアクセスする
    2.x[0]のサイズが5 * 5 * 5 = 125
    3 tf.size(x[0]) = 125 となる

    サイズ取得のみが目的なので、x[0]は0でなくても構わない(範囲外アクセスにならなければ)
    """
    size = tf.size(x[0])
    #-1とする事で、バッチサイズを動的に取得してくれる
    return tf.reshape(x, [-1, size])

def GAP(x):
    for _ in range(2):
        x = tf.reduce_mean(x, axis=1)
    return x

def softmax(array):
    if array.ndim == 2:
        array = array.T
        array = array - np.max(array, axis=0)
        y = np.exp(array) / np.sum(np.exp(array), axis=0)
        return y.T
    array = array - np.max(array) # オーバーフロー対策
    return np.exp(array) / np.sum(np.exp(array))

def accuracy(output, target_label):
    correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(target_label, 1))
    return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

def mean_squared_error(output, target_label):
    reduce_sum = tf.reduce_sum((output - target_label) ** 2, axis=1)
    while True:
        if len(reduce_sum.shape) == 1:
            break
        reduce_sum = tf.reduce_sum(reduce_sum, axis=1)
    return tf.reduce_mean(reduce_sum)

def cross_entropy(output, target_label):
    clip_output = tf.clip_by_value(output, 1e-10, 1.0)
    return tf.reduce_mean(-tf.reduce_sum(target_label * tf.log(clip_output), axis=1))

def upper_confidence_bound1(average_value, total_trial, trial, X):
    return average_value + (X * math.sqrt(2 * math.log(total_trial) / trial))

#gammaは割引率
def updated_q(q_array, next_q_array, reward, learning_rate, gamma):
    return q_array + learning_rate * (reward + gamma * max(next_q_array) - q_array)

def boltzmann_distribution(array, temperature_parameter):
    return array ** (1 / temperature_parameter) / np.sum(array ** (1 / temperature_parameter))

def sigmoid_y_to_tanh_y(sigmoid_y):
    assert sigmoid_y >= 0.0 and sigmoid_y <= 1.0
    return (2 * sigmoid_y) - 1

def tanh_y_to_sigmoid_y(tanh_y):
    assert tanh_y >= -1.0 and tanh_y <= 1.0
    return (0.5 * tanh_y) + 0.5

def argmaxes(data):
    max_ = max(data)
    return [i for i, ele in enumerate(data) if ele == max_]

def epsilon_greedy(data, random_percent):
    assert random_percent >= 0 and random_percent <= 1.0
    if random.random() > random_percent:
        return max_index_random_choice(data)
    return random.randint(0, len(data) - 1)

def boltzmann_random(array, temperature_parameter):
    if temperature_parameter == -1:
        return random.randint(0, len(array) - 1)

    if temperature_parameter == 0:
        return max_index_random_choice(array)

    if all(array == 0):
        return random.randint(0, len(array) - 1)

    assert temperature_parameter > 0
    assert all(array >= 0)

    boltzmann_array = boltzmann_distribution(array, temperature_parameter)
    threshold = random.uniform(0.0, boltzmann_array.sum())
    tmp_sum = 0

    for i, value in enumerate(boltzmann_array):
        tmp_sum += value
        if tmp_sum > threshold:
            return i
    assert False, "たぶんおそらくもしかしたら計算途中でNonになっている可能性が高いかもしれないと思うよ"

def load_mnist(flatten, binary):
    (train_data, target_data), (test_data, test_target) = \
        mnist.load_mnist(flatten=flatten, one_hot_label=True, normalize=True)

    if binary:
        train_data = np.where(train_data != 0, 1, 0)
        test_data = np.where(test_data != 0, 1, 0)

    return train_data, target_data, test_data, test_target

class AdaBoundOptimizer(optimizer.Optimizer):
    def __init__(self, learning_rate=0.001, final_lr=0.1, beta1=0.9, beta2=0.999,
                 gamma=1e-3, epsilon=1e-8, amsbound=False,
                 use_locking=False, name="AdaBound"):
        super(AdaBoundOptimizer, self).__init__(use_locking, name)
        self._lr = learning_rate
        self._final_lr = final_lr
        self._beta1 = beta1
        self._beta2 = beta2
        self._epsilon = epsilon

        self._gamma = gamma
        self._amsbound = amsbound

        self._lr_t = None
        self._beta1_t = None
        self._beta2_t = None
        self._epsilon_t = None

    def _create_slots(self, var_list):
        first_var = min(var_list, key=lambda x: x.name)

        graph = None if context.executing_eagerly() else ops.get_default_graph()
        create_new = self._get_non_slot_variable("beta1_power", graph) is None
        if not create_new and context.in_graph_mode():
            create_new = (self._get_non_slot_variable("beta1_power", graph).graph is not first_var.graph)

        if create_new:
            self._create_non_slot_variable(initial_value=self._beta1,
                                           name="beta1_power",
                                           colocate_with=first_var)
            self._create_non_slot_variable(initial_value=self._beta2,
                                           name="beta2_power",
                                           colocate_with=first_var)
            self._create_non_slot_variable(initial_value=self._gamma,
                                           name="gamma_multi",
                                           colocate_with=first_var)
        # Create slots for the first and second moments.
        for v in var_list :
            self._zeros_slot(v, "m", self._name)
            self._zeros_slot(v, "v", self._name)
            self._zeros_slot(v, "vhat", self._name)

    def _prepare(self):
        self._lr_t = ops.convert_to_tensor(self._lr)
        self._base_lr_t = ops.convert_to_tensor(self._lr)
        self._beta1_t = ops.convert_to_tensor(self._beta1)
        self._beta2_t = ops.convert_to_tensor(self._beta2)
        self._epsilon_t = ops.convert_to_tensor(self._epsilon)
        self._gamma_t = ops.convert_to_tensor(self._gamma)

    def _apply_dense(self, grad, var):
        graph = None if context.executing_eagerly() else ops.get_default_graph()
        beta1_power = math_ops.cast(self._get_non_slot_variable("beta1_power", graph=graph), var.dtype.base_dtype)
        beta2_power = math_ops.cast(self._get_non_slot_variable("beta2_power", graph=graph), var.dtype.base_dtype)
        lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)
        base_lr_t = math_ops.cast(self._base_lr_t, var.dtype.base_dtype)
        beta1_t = math_ops.cast(self._beta1_t, var.dtype.base_dtype)
        beta2_t = math_ops.cast(self._beta2_t, var.dtype.base_dtype)
        epsilon_t = math_ops.cast(self._epsilon_t, var.dtype.base_dtype)
        gamma_multi = math_ops.cast(self._get_non_slot_variable("gamma_multi", graph=graph), var.dtype.base_dtype)

        step_size = (lr_t * math_ops.sqrt(1 - beta2_power) / (1 - beta1_power))
        final_lr = self._final_lr * lr_t / base_lr_t
        lower_bound = final_lr * (1. - 1. / (gamma_multi + 1.))
        upper_bound = final_lr * (1. + 1. / (gamma_multi))

        # m_t = beta1 * m + (1 - beta1) * g_t
        m = self.get_slot(var, "m")
        m_scaled_g_values = grad * (1 - beta1_t)
        m_t = state_ops.assign(m, beta1_t * m + m_scaled_g_values, use_locking=self._use_locking)

        # v_t = beta2 * v + (1 - beta2) * (g_t * g_t)
        v = self.get_slot(var, "v")
        v_scaled_g_values = (grad * grad) * (1 - beta2_t)
        v_t = state_ops.assign(v, beta2_t * v + v_scaled_g_values, use_locking=self._use_locking)

        # amsgrad
        vhat = self.get_slot(var, "vhat")
        if self._amsbound :
            vhat_t = state_ops.assign(vhat, math_ops.maximum(v_t, vhat))
            v_sqrt = math_ops.sqrt(vhat_t)
        else :
            vhat_t = state_ops.assign(vhat, vhat)
            v_sqrt = math_ops.sqrt(v_t)


        # Compute the bounds
        step_size_bound = step_size / (v_sqrt + epsilon_t)
        bounded_lr = m_t * clip_by_value(step_size_bound, lower_bound, upper_bound)

        var_update = state_ops.assign_sub(var, bounded_lr, use_locking=self._use_locking)
        return control_flow_ops.group(*[var_update, m_t, v_t, vhat_t])

    def _resource_apply_dense(self, grad, var):
        graph = None if context.executing_eagerly() else ops.get_default_graph()
        beta1_power = math_ops.cast(self._get_non_slot_variable("beta1_power", graph=graph), grad.dtype.base_dtype)
        beta2_power = math_ops.cast(self._get_non_slot_variable("beta2_power", graph=graph), grad.dtype.base_dtype)
        lr_t = math_ops.cast(self._lr_t, grad.dtype.base_dtype)
        base_lr_t = math_ops.cast(self._base_lr_t, var.dtype.base_dtype)
        beta1_t = math_ops.cast(self._beta1_t, grad.dtype.base_dtype)
        beta2_t = math_ops.cast(self._beta2_t, grad.dtype.base_dtype)
        epsilon_t = math_ops.cast(self._epsilon_t, grad.dtype.base_dtype)
        gamma_multi = math_ops.cast(self._get_non_slot_variable("gamma_multi", graph=graph), var.dtype.base_dtype)

        step_size = (lr_t * math_ops.sqrt(1 - beta2_power) / (1 - beta1_power))
        final_lr = self._final_lr * lr_t / base_lr_t
        lower_bound = final_lr * (1. - 1. / (gamma_multi + 1.))
        upper_bound = final_lr * (1. + 1. / (gamma_multi))

        # m_t = beta1 * m + (1 - beta1) * g_t
        m = self.get_slot(var, "m")
        m_scaled_g_values = grad * (1 - beta1_t)
        m_t = state_ops.assign(m, beta1_t * m + m_scaled_g_values, use_locking=self._use_locking)

        # v_t = beta2 * v + (1 - beta2) * (g_t * g_t)
        v = self.get_slot(var, "v")
        v_scaled_g_values = (grad * grad) * (1 - beta2_t)
        v_t = state_ops.assign(v, beta2_t * v + v_scaled_g_values, use_locking=self._use_locking)

        # amsgrad
        vhat = self.get_slot(var, "vhat")
        if self._amsbound:
            vhat_t = state_ops.assign(vhat, math_ops.maximum(v_t, vhat))
            v_sqrt = math_ops.sqrt(vhat_t)
        else:
            vhat_t = state_ops.assign(vhat, vhat)
            v_sqrt = math_ops.sqrt(v_t)

        # Compute the bounds
        step_size_bound = step_size / (v_sqrt + epsilon_t)
        bounded_lr = m_t * clip_by_value(step_size_bound, lower_bound, upper_bound)

        var_update = state_ops.assign_sub(var, bounded_lr, use_locking=self._use_locking)

        return control_flow_ops.group(*[var_update, m_t, v_t, vhat_t])

    def _apply_sparse_shared(self, grad, var, indices, scatter_add):
        graph = None if context.executing_eagerly() else ops.get_default_graph()
        beta1_power = math_ops.cast(self._get_non_slot_variable("beta1_power", graph=graph), var.dtype.base_dtype)
        beta2_power = math_ops.cast(self._get_non_slot_variable("beta2_power", graph=graph), var.dtype.base_dtype)
        lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)
        base_lr_t = math_ops.cast(self._base_lr_t, var.dtype.base_dtype)
        beta1_t = math_ops.cast(self._beta1_t, var.dtype.base_dtype)
        beta2_t = math_ops.cast(self._beta2_t, var.dtype.base_dtype)
        epsilon_t = math_ops.cast(self._epsilon_t, var.dtype.base_dtype)
        gamma_t = math_ops.cast(self._gamma_t, var.dtype.base_dtype)

        step_size = (lr_t * math_ops.sqrt(1 - beta2_power) / (1 - beta1_power))
        final_lr = self._final_lr * lr_t / base_lr_t
        lower_bound = final_lr * (1. - 1. / (gamma_t + 1.))
        upper_bound = final_lr * (1. + 1. / (gamma_t))

        # m_t = beta1 * m + (1 - beta1) * g_t
        m = self.get_slot(var, "m")
        m_scaled_g_values = grad * (1 - beta1_t)
        m_t = state_ops.assign(m, m * beta1_t, use_locking=self._use_locking)
        with ops.control_dependencies([m_t]):
            m_t = scatter_add(m, indices, m_scaled_g_values)

        # v_t = beta2 * v + (1 - beta2) * (g_t * g_t)
        v = self.get_slot(var, "v")
        v_scaled_g_values = (grad * grad) * (1 - beta2_t)
        v_t = state_ops.assign(v, v * beta2_t, use_locking=self._use_locking)
        with ops.control_dependencies([v_t]):
            v_t = scatter_add(v, indices, v_scaled_g_values)

        # amsgrad
        vhat = self.get_slot(var, "vhat")
        if self._amsbound:
            vhat_t = state_ops.assign(vhat, math_ops.maximum(v_t, vhat))
            v_sqrt = math_ops.sqrt(vhat_t)
        else:
            vhat_t = state_ops.assign(vhat, vhat)
            v_sqrt = math_ops.sqrt(v_t)

        # Compute the bounds
        step_size_bound = step_size / (v_sqrt + epsilon_t)
        bounded_lr = m_t * clip_by_value(step_size_bound, lower_bound, upper_bound)

        var_update = state_ops.assign_sub(var, bounded_lr, use_locking=self._use_locking)

        return control_flow_ops.group(*[var_update, m_t, v_t, vhat_t])

    def _apply_sparse(self, grad, var):
        return self._apply_sparse_shared(
            grad.values, var, grad.indices,
            lambda x, i, v: state_ops.scatter_add(  # pylint: disable=g-long-lambda
                x, i, v, use_locking=self._use_locking))

    def _resource_scatter_add(self, x, i, v):
        with ops.control_dependencies(
                [resource_variable_ops.resource_scatter_add(x, i, v)]):
            return x.value()

    def _resource_apply_sparse(self, grad, var, indices):
        return self._apply_sparse_shared(
            grad, var, indices, self._resource_scatter_add)

    def _finish(self, update_ops, name_scope):
        # Update the power accumulators.
        with ops.control_dependencies(update_ops):
            graph = None if context.executing_eagerly() else ops.get_default_graph()
            beta1_power = self._get_non_slot_variable("beta1_power", graph=graph)
            beta2_power = self._get_non_slot_variable("beta2_power", graph=graph)
            gamma_multi = self._get_non_slot_variable("gamma_multi", graph=graph)
            with ops.colocate_with(beta1_power):
                update_beta1 = beta1_power.assign(
                    beta1_power * self._beta1_t,
                    use_locking=self._use_locking)
                update_beta2 = beta2_power.assign(
                    beta2_power * self._beta2_t,
                    use_locking=self._use_locking)
                update_gamma = gamma_multi.assign(
                    gamma_multi + self._gamma_t,
                    use_locking=self._use_locking)
        return control_flow_ops.group(*update_ops + [update_beta1, update_beta2, update_gamma],
                                      name=name_scope)

if __name__ == "__main__":
    q_array = np.array([0.3, 0.2, 0.5, 0.3])
    next_q_array = np.array([0,5, 0.4, 0.3, 0.2])
    print(updated_q(q_array, next_q_array, 1.0, 0.001, 1.0))
