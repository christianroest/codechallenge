from data import make_tf_dataset

dataset = make_tf_dataset('/mnt/d/code_data')
for imgs, segs in dataset.take(5):
  print(imgs.numpy().shape)
  print(segs.numpy().shape)