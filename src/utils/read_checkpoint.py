import tensorflow as tf
import params

CONFIGS = params.load_configs()

ckpt_reader = open(CONFIGS.ckpt_reader, "w")

reader = tf.train.NewCheckpointReader(CONFIGS.ckpt_path)
var_to_shape_map = reader.get_variable_to_shape_map()
for key in var_to_shape_map:
    print("tensor_name %s" % key, file=ckpt_reader)

# from tensorflow.python.tools import inspect_checkpoint as chkp

# # print all tensors in checkpoint file
# chkp.print_tensors_in_checkpoint_file(CONFIGS.ckpt_path, tensor_name='', all_tensors=True)
