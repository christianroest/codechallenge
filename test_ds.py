import tensorflow as tf

l = tf.data.Dataset.list_files("/mnt/d/code_data/video_01/rgb/*.png", shuffle=False)

for f in l.take(5):
  print(f.numpy().shape)