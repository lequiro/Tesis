import tensorflow as tf


class RK4Model(tf.keras.Model):
    def __init__(self, base_model):
        super(RK4Model, self).__init__()
        self.base_model = base_model

    def call(self, inputs):
        return self.base_model(inputs)

    @tf.function
    def rk4_nn(self, state, dt):
        k1 = dt * self(state)
        k2 = dt * self(state + 0.5 * k1)
        k3 = dt * self(state + 0.5 * k2)
        k4 = dt * self(state + k3)
        next_state = state + (k1 + 2 * k2 + 2 * k3 + k4) / 6
        return next_state

    @tf.function
    def forward_rk4_nn(self, initial_states, num_steps_batch, dt):
        max_num_steps = tf.reduce_max(num_steps_batch)
        final_states = initial_states

        i = tf.constant(0)

        def condition(i, final_states):
            return i < max_num_steps

        def body(i, final_states):
            mask = num_steps_batch > i  # shape: [batch_size]
            indices = tf.where(mask)
            states_to_update = tf.gather_nd(final_states, indices)
            next_states = self.rk4_nn(states_to_update, dt)
            final_states = tf.tensor_scatter_nd_update(final_states, indices, next_states)
            return i + 1, final_states

        _, final_states = tf.while_loop(condition, body, [i, final_states])
        return final_states

    @tf.function
    def forward_rk4_nn_all(self, initial_states, num_steps_batch, dt):
        # This function stores all intermediate states.
        max_num_steps = tf.reduce_max(num_steps_batch)
        final_states = initial_states

        # Create a TensorArray to store states at each step
        states_ta = tf.TensorArray(dtype=initial_states.dtype, size=max_num_steps + 1)
        states_ta = states_ta.write(0, initial_states)

        i = tf.constant(0)

        def condition(i, final_states, states_ta):
            return i < max_num_steps

        def body(i, final_states, states_ta):
            mask = num_steps_batch > i
            indices = tf.where(mask)
            states_to_update = tf.gather_nd(final_states, indices)
            next_states = self.rk4_nn(states_to_update, dt)
            final_states = tf.tensor_scatter_nd_update(final_states, indices, next_states)
            states_ta = states_ta.write(i + 1, final_states)
            return i + 1, final_states, states_ta

        _, final_states, states_ta = tf.while_loop(condition, body, [i, final_states, states_ta])
        all_states = states_ta.stack()
        return final_states, all_states
