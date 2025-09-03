import tensorflow as tf
from tensorflow.keras.losses import Loss

class CategoricalFocalCrossentropy(Loss):
    def __init__(self, gamma=2.0, alpha=0.25, from_logits=False, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.alpha = alpha
        self.from_logits = from_logits

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        if self.from_logits:
            y_pred = tf.nn.softmax(y_pred, axis=-1)
        else:
            y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())

        cross_entropy = -y_true * tf.math.log(y_pred)
        weights = self.alpha * tf.pow(1 - y_pred, self.gamma)
        loss = weights * cross_entropy
        return tf.reduce_sum(loss, axis=-1)
        
class CategoricalDiceLoss(Loss):
    def __init__(self, smooth=1e-6, from_logits=False, **kwargs):
        super().__init__(**kwargs)
        self.smooth = smooth
        self.from_logits = from_logits

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        if self.from_logits:
            y_pred = tf.nn.softmax(y_pred, axis=-1)
        else:
            y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1.0)

        intersection = tf.reduce_sum(y_true * y_pred, axis=[1,2])
        union = tf.reduce_sum(y_true + y_pred, axis=[1,2])
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - tf.reduce_mean(dice, axis=-1)