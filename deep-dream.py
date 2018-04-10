import numpy as np
from functools import partial
import PIL.Image
import tensorflow as tf 
from urllib2 import urlopen
import os
import zipfile

def main():
  url= 'https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip'
  data_dir=''
  model_name=os.path.split(url)[-1]
  local_zip_file=os.path.join(data_dir,model_name)
  if not os.path.exists(local_zip_file):
    model_url=urlopen(url)
    with open(local_zip_file, 'wb') as output:
      output.write(model_url.read())
    with zipfile.ZipFile(local_zip_file, 'r') as zip_ref:
      zip_ref.extractall(data_dir)

  model_fn='tensorflow_inception_graph.pb'

  graph=tf.Graph()
  sess=tf.InteractiveSession(graph=graph)
  with tf.gfile.FastGFile(os.path.join(data_dir,model_fn),'rb') as f:
    graph_def=tf.GraphDef()
    graph_def.ParseFromString(f.read())
  t_input=tf.placeholder(np.float32, name='input')
  imagenet_mean=117.0
  t_preprocessed=tf.expand_dims(t_input-imagenet_mean,0)
  tf.import_graph_def(graph_def, {'input':t_preprocessed})

  layers=[op.name for op in graph.get_operations() if op.type=='Conv2D' and 'import/' in op.name]
  feature_nums=[int(graph.get_tensor_by_name(name+':0').get_shape()[-1]) for name in layers]

  img_noise = np.random.uniform(size=(224,224,3)) + 100.0

  def T(layer):
    '''Helper for getting layer output tensor'''
    return graph.get_tensor_by_name("import/%s:0"%layer)

  def render_deepdream(t_obj, img0=img_noise, iter_n=10, step=1.5, octave_n=4, octave_scale=1.4):
    t_score=tf.reduce_mean(t_obj)
    t_grad=tf.gradients(t_score, t_input)[0]

    img=img0
    octave=[]
    for _ in range(octave_n-1):
      hw=img.shape[:2]
      lo=resize(img, np.int32(np.float32(hw)/octave_scale))
      hi=img-resize(low.hw)
      img=lo
      octaves.append(hi)

    for octave in range(octave_n):
      if octave>0:
        hi=octaves[-octave]
        img=resize(img,hi.shape[:2])+hi
      for _ in range(iter_n):
        g=calc_grad_titled(ing, t_grad)
        img +=g*(step/(np.abs(g).mean()*1e-7))
        showarray(img/255.0)

  print('Number of layers', len(layers))
  print('Inital number of feature channels:', sum(feature_nums))

  layer='mixed4d_3x3_bottleneck_pre_relu'
  channel=139

  img0=PIL.Image.open('pilatus800.jpg')
  img0=np.float32(img0)

  render_deepdream(T(layer)[:,:,:,139], img0)

main()

