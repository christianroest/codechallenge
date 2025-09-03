import tensorflow as tf

class SparseDiceLoss(tf.keras.losses.Loss):
    def __init__(self, num_classes, ignore_index=0, smooth=1e-6, name="sparse_dice_loss"):
        super().__init__(name=name)
        self.ignore_index = ignore_index
        self.smooth = smooth
        self.num_classes = num_classes

    def call(self, y_true, y_pred):
        """
        y_true: [B, H, W] int32 class indices
        y_pred: [B, H, W, C] logits (not softmaxed)
        """
        self.num_classes = tf.shape(y_pred)[-1]
        probs = tf.nn.softmax(y_pred, axis=-1)

        y_true = tf.cast(y_true, tf.int32)
        y_true_onehot = tf.one_hot(y_true, depth=self.num_classes, dtype=tf.float32)

        # [B, H, W, C] -> flatten spatial dims
        probs = tf.reshape(probs, [-1, self.num_classes])
        y_true_onehot = tf.reshape(y_true_onehot, [-1, self.num_classes])

        intersection = tf.reduce_sum(probs * y_true_onehot, axis=0)
        denom = tf.reduce_sum(probs, axis=0) + tf.reduce_sum(y_true_onehot, axis=0)
        dice = (2.0 * intersection + self.smooth) / (denom + self.smooth)

        # Mask out background (ignore_index)
        mask = tf.ones_like(dice, dtype=tf.bool)
        mask = tf.tensor_scatter_nd_update(
            mask, [[self.ignore_index]], [False]
        )

        losses = 1.0 - dice
        losses = tf.boolean_mask(losses, mask)

        return tf.reduce_mean(losses)