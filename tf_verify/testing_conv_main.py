import sys
import os
cpu_affinity = os.sched_getaffinity(0)
sys.path.insert(0, '../ELINA/python_interface/')
sys.path.insert(0, '../deepg/code/')
import numpy as np
from eran import ERAN
from read_net_file import *
import tensorflow as tf
import csv
import random
import cv2
import time
from tqdm import tqdm
from ai_milp import *
import argparse
from config import config
from constraint_utils import *
import re
import itertools
from multiprocessing import Pool, Value
import onnxruntime.backend as rt
import logging
#ZONOTOPE_EXTENSION = '.zt'
EPS = 10**(-9)

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def isnetworkfile(fname):
    _, ext = os.path.splitext(fname)
    if ext not in ['.pyt', '.meta', '.tf','.onnx', '.pb']:
        raise argparse.ArgumentTypeError('only .pyt, .tf, .onnx, .pb, and .meta formats supported')
    return fname

def normalize(image, means, stds, dataset):
    # normalization taken out of the network
    if len(means) == len(image):
        for i in range(len(image)):
            image[i] -= means[i]
            if stds!=None:
                image[i] /= stds[i]
    elif dataset == 'mnist'  or dataset == 'fashion':
        for i in range(len(image)):
            image[i] = (image[i] - means[0])/stds[0]
    elif(dataset=='cifar10'):
        count = 0
        tmp = np.zeros(3072)
        for i in range(1024):
            tmp[count] = (image[count] - means[0])/stds[0]
            count = count + 1
            tmp[count] = (image[count] - means[1])/stds[1]
            count = count + 1
            tmp[count] = (image[count] - means[2])/stds[2]
            count = count + 1

        if(is_conv):
            for i in range(3072):
                image[i] = tmp[i]
        else:
            count = 0
            for i in range(1024):
                image[i] = tmp[count]
                count = count+1
                image[i+1024] = tmp[count]
                count = count+1
                image[i+2048] = tmp[count]
                count = count+1

def parse_input_box(text):
    intervals_list = []
    for line in text.split('\n'):
        if line!="":
            interval_strings = re.findall("\[-?\d*\.?\d+, *-?\d*\.?\d+\]", line)
            intervals = []
            for interval in interval_strings:
                interval = interval.replace('[', '')
                interval = interval.replace(']', '')
                [lb,ub] = interval.split(",")
                intervals.append((np.double(lb), np.double(ub)))
            intervals_list.append(intervals)

    # return every combination
    boxes = itertools.product(*intervals_list)
    return boxes

def show_ascii_spec(lb, ub, n_rows, n_cols, n_channels):
    print('==================================================================')
    for i in range(n_rows):
        print('  ', end='')
        for j in range(n_cols):
            print('#' if lb[n_cols*n_channels*i+j*n_channels] >= 0.5 else ' ', end='')
        print('  |  ', end='')
        for j in range(n_cols):
            print('#' if ub[n_cols*n_channels*i+j*n_channels] >= 0.5 else ' ', end='')
        print('  |  ')
    print('==================================================================')

def normalize_poly(num_params, lexpr_cst, lexpr_weights, lexpr_dim, uexpr_cst, uexpr_weights, uexpr_dim, means, stds, dataset):
    # normalization taken out of the network
    if dataset == 'mnist' or dataset == 'fashion':
        for i in range(len(lexpr_cst)):
            lexpr_cst[i] = (lexpr_cst[i] - means[0]) / stds[0]
            uexpr_cst[i] = (uexpr_cst[i] - means[0]) / stds[0]
        for i in range(len(lexpr_weights)):
            lexpr_weights[i] /= stds[0]
            uexpr_weights[i] /= stds[0]
    else:
        for i in range(len(lexpr_cst)):
            lexpr_cst[i] = (lexpr_cst[i] - means[i % 3]) / stds[i % 3]
            uexpr_cst[i] = (uexpr_cst[i] - means[i % 3]) / stds[i % 3]
        for i in range(len(lexpr_weights)):
            lexpr_weights[i] /= stds[(i // num_params) % 3]
            uexpr_weights[i] /= stds[(i // num_params) % 3]

def denormalize(image, means, stds, dataset):
    if dataset == 'mnist'  or dataset == 'fashion':
        for i in range(len(image)):
            image[i] = image[i]*stds[0] + means[0]
    elif(dataset=='cifar10'):
        count = 0
        tmp = np.zeros(3072)
        for i in range(1024):
            tmp[count] = image[count]*stds[0] + means[0]
            count = count + 1
            tmp[count] = image[count]*stds[1] + means[1]
            count = count + 1
            tmp[count] = image[count]*stds[2] + means[2]
            count = count + 1

        for i in range(3072):
            image[i] = tmp[i]

def upsamplefeatures(input):
    N, Cin, Hin, Win = input.shape
    sca = 2
    sca2 = sca*sca
    Cout = Cin//sca2
    Hout = Hin*sca
    Wout = Win*sca
    idxL = [[0, 0], [0, 1], [1, 0], [1, 1]]
    assert (Cin%sca2 == 0), 'Invalid input dimensions: number of channels should be divisible by 4'
    result = np.zeros((N, Cout, Hout, Wout), dtype=np.float32)
    for idx in range(sca2):
        result[:, :, idxL[idx][0]::sca, idxL[idx][1]::sca] = input[:, idx:Cin:sca2, :, :]
    return result

def get_tests(dataset, geometric):
    if geometric:
        csvfile = open('../deepg/code/datasets/{}_test.csv'.format(dataset), 'r')
    else:
        csvfile = open('../data/{}_test.csv'.format(dataset), 'r')
    tests = csv.reader(csvfile, delimiter=',')
    return tests

def init_domain(d):
    if d == 'refinezono':
        return 'deepzono'
    elif d == 'refinepoly':
        return 'deeppoly'
    else:
        return d

def variable_to_cv2_image(varim):
    res = (varim * 255.).clip(0, 255).astype(np.uint8)	
    return res

def normalize_ffd(data):
    """Normalizes a unit8 image to a float32 image in the range [0, 1]

	Args:
		data: a unint8 numpy array to normalize from [0, 255] to [0, 1]
		The normalization function from ffdnet
	"""
    return np.float32(data/255.)

def concatenate_input_noise_map(input, noise_sigma):
    N, C, H, W = input.shape
    sca = 2
    sca2 = sca*sca
    Cout = sca2*C
    Hout = H//sca
    Wout = W//sca
    idxL = [[0, 0], [0, 1], [1, 0], [1, 1]]
    downsampledfeatures = np.zeros((N, Cout, Hout, Wout), dtype=np.float32)
    # The N above equals to 1
    noise_map = np.full((1, C, Hout, Wout), noise_sigma, dtype=np.float32)
    print(downsampledfeatures.shape)
    print(noise_map.shape)
    for idx in range(sca2):
        downsampledfeatures[:, idx:Cout:sca2, :, :] = input[:, :, idxL[idx][0]::sca, idxL[idx][1]::sca]
    return np.concatenate((noise_map, downsampledfeatures), axis=1)

def segment_network(model):
    layer_list = list(model.intermediate_dncnn.children())[0][:2]

parser = argparse.ArgumentParser(description='ERAN Example',  formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--netname', type=isnetworkfile, default=config.netname, help='the network name, the extension can be only .pb, .pyt, .tf, .meta, and .onnx')
parser.add_argument('--epsilon', type=float, default=config.epsilon, help='the epsilon for L_infinity perturbation')
parser.add_argument('--dataset', type=str, default=config.dataset, help='the dataset, can be either mnist, cifar10')
############## My parser
parser.add_argument('--layer_by_layer', type=str2bool, default=False,  help='Will the back-substitution be conducted layer by layer?')
parser.add_argument('--is_residual', type=str2bool, default=False,  help='whether the verification network is residual network or not')
parser.add_argument('--blk_size', type=int, default=0,  help='How many affine layers are contained in one block')
parser.add_argument('--is_blk_segmentation', type=str2bool, default=False,  help='flag to allow modular analysis in blocks')
parser.add_argument('--is_early_terminate', type=str2bool, default=False,  help='flag to allow early termination or not')
parser.add_argument('--early_termi_thre', type=int, default=0,  help='threshold for early termination')
parser.add_argument('--is_sum_def_over_input', type=str2bool, default=False,  help='if the summary is input summary or not (if not, it will be block summary')
parser.add_argument('--stage_layer_num', type=int, default=None,  help='The number of how many layers in one block during segmentation, the default is None meaning that there is no staging included')
parser.add_argument('--stage_block_num', type=int, default=None,  help='The number of how many block we will have after segmentation, the default is None')
############ End of my parserssss
parser.add_argument('--use_default_heuristic', type=str2bool, default=config.use_default_heuristic,  help='whether to use the area heuristic for the DeepPoly ReLU approximation or to always create new noise symbols per relu for the DeepZono ReLU approximation')
parser.add_argument('--mean', nargs='+', type=float, default=config.mean, help='the mean used to normalize the data with')
parser.add_argument('--std', nargs='+', type=float, default=config.std, help='the standard deviation used to normalize the data with')


# Logging options
parser.add_argument('--logdir', type=str, default=None, help='Location to save logs to. If not specified, logs are not saved and emitted to stdout')
parser.add_argument('--logname', type=str, default=None, help='Directory of log files in `logdir`, if not specified timestamp is used')


args = parser.parse_args() 
# The return value from parse_args() is a Namespace containing the arguments to the command. The object holds the argument values as attributes
for k, v in vars(args).items():
    setattr(config, k, v) #takes three parameters:object whose attributes to be set, attribute name, attribute name
config.json = vars(args)

assert config.netname, 'a network has to be provided for analysis.'

netname = config.netname
filename, file_extension = os.path.splitext(netname)

is_trained_with_pytorch = file_extension==".pyt"
is_saved_tf_model = file_extension==".meta"
is_pb_file = file_extension==".pb"
is_tensorflow = file_extension== ".tf"
is_onnx = file_extension == ".onnx"
assert is_trained_with_pytorch or is_saved_tf_model or is_pb_file or is_tensorflow or is_onnx, "file extension not supported"

epsilon = config.epsilon
assert (epsilon >= 0) and (epsilon <= 1), "epsilon can only be between 0 and 1"
dataset = config.dataset
assert dataset in ['mnist', 'cifar10'], "only mnist, cifar10 datasets are supported"
mean = 0
std = 0
is_conv = False
print("netname ", netname, " epsilon ", epsilon, " domain ", 'deeppoly', " dataset ", dataset)

non_layer_operation_types = ['NoOp', 'Assign', 'Const', 'RestoreV2', 'SaveV2', 'PlaceholderWithDefault', 'IsVariableInitialized', 'Placeholder', 'Identity']

if is_saved_tf_model or is_pb_file:
    netfolder = os.path.dirname(netname)

    tf.logging.set_verbosity(tf.logging.ERROR)

    sess = tf.Session()
    if is_saved_tf_model:
        saver = tf.train.import_meta_graph(netname)
        saver.restore(sess, tf.train.latest_checkpoint(netfolder+'/'))
    else:
        #Start from sess = tf.Session()
        with tf.gfile.GFile(netname, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            sess.graph.as_default()
            tf.graph_util.import_graph_def(graph_def, name='')
        #End in here, are just normal codes that are used to load pb file
    ops = sess.graph.get_operations()
    last_layer_index = -1
    while ops[last_layer_index].type in non_layer_operation_types:
        last_layer_index -= 1
    eran = ERAN(sess.graph.get_tensor_by_name(ops[last_layer_index].name + ':0'), sess)
else:
    if(dataset=='mnist'):
        num_pixels = 784
    elif (dataset=='cifar10'):
        num_pixels = 3072
    if is_onnx:
        model, is_conv = read_onnx_net(netname)
        # this is to have different defaults for mnist and cifar10
    else:
        model, is_conv, means, stds = read_tensorflow_net(netname, num_pixels, is_trained_with_pytorch)
    eran = ERAN(model, is_onnx=is_onnx)

if not is_trained_with_pytorch:
    if dataset == 'mnist':
        means = [0]
        stds = [1]
    else:
        means = [0.5, 0.5, 0.5]
        stds = [1, 1, 1]

is_trained_with_pytorch = is_trained_with_pytorch or is_onnx

if config.mean is not None:
    means = config.mean
    stds = config.std

if dataset:
    tests = get_tests(dataset, config.geometric)

net_name_list = netname.split("/")
net_file = net_name_list[-1]
specLB = None
specUB = None
epsilon_list_conv = None
if dataset == 'mnist':
    epsilon_list_conv = [0.02, 0.04, 0.06, 0.08, 0.1, 0.12]
    # epsilon_list_conv = [0.12, 0.14, 0.16, 0.18, 0.20, 0.22]
    
else:
    epsilon_list_conv = [0.002, 0.004, 0.006, 0.008, 0.01, 0.012]
    
dp_count = 0
blksum_count = 0
inpsum_count = 0
candi_count = 0

for i, test in enumerate(tests):
    image= np.float64(test[1:len(test)])/np.float64(255)
    actual_label= int(test[0])
    specLB = np.copy(image)
    specUB = np.copy(image)
    normalize(specLB, means, stds, dataset)
    normalize(specUB, means, stds, dataset)
    eran_result = eran.analyze_box(specLB, specUB, init_domain('deeppoly'), config.timeout_lp, config.timeout_milp, config.use_default_heuristic)
    dominant_class = eran_result[0]
    if(dominant_class == actual_label):
        candi_count = candi_count + 1
        for epsilon in epsilon_list_conv:
            if config.normalized_region==True:
                specLB = np.clip(image - epsilon,0,1)
                specUB = np.clip(image + epsilon,0,1)
                normalize(specLB, means, stds, dataset)
                normalize(specUB, means, stds, dataset)
            else:
                specLB = specLB - epsilon
                specUB = specUB + epsilon


            lb_fullpath = "stress_test_result/labels_" + net_file.replace(".", "_") +".csv"

            # The old deeppoly execution 
            start = time.time() 
            eran_result = eran.analyze_box(specLB, specUB, init_domain('deeppoly'), config.timeout_lp, config.timeout_milp, config.use_default_heuristic)
            end = time.time()
            dominant_class = eran_result[0]
            if(dominant_class == actual_label):
                dp_count = dp_count + 1
            with open(lb_fullpath, 'a+', newline='') as write_obj:
                csv_writer = csv.writer(write_obj)
                csv_writer.writerow([net_file, str(dataset), "img "+str(i)+" with label "+str(actual_label), "eps="+str(epsilon), "deeppoly", str(end - start)+" secs", str(dominant_class)])
        
            # The execution of staging
            start = time.time()
            eran_result = eran.analyze_box(specLB, specUB, init_domain('deeppoly'), config.timeout_lp, config.timeout_milp, config.use_default_heuristic, False, False, True, 3, False, False, False, False)
            end = time.time()
            dominant_class = eran_result[0]
            if(dominant_class == actual_label):
                blksum_count = blksum_count + 1
            with open(lb_fullpath, 'a+', newline='') as write_obj:
                csv_writer = csv.writer(write_obj)
                csv_writer.writerow([net_file, str(dataset), "img "+str(i)+" with label "+str(actual_label), "eps="+str(epsilon), "blk3_blksum", str(end - start)+" secs", str(dominant_class)])

            start = time.time()
            eran_result = eran.analyze_box(specLB, specUB, init_domain('deeppoly'), config.timeout_lp, config.timeout_milp, config.use_default_heuristic, False, False, True, 3, False, False, True, False)
            end = time.time()
            dominant_class = eran_result[0]
            if(dominant_class == actual_label):
                inpsum_count = inpsum_count + 1
            with open(lb_fullpath, 'a+', newline='') as write_obj:
                csv_writer = csv.writer(write_obj)
                csv_writer.writerow([net_file, str(dataset), "img "+str(i)+" with label "+str(actual_label), "eps="+str(epsilon), "blk3_inputsum", str(end - start)+" secs", str(dominant_class)])
                
print(candi_count, dp_count, blksum_count, inpsum_count)
with open(lb_fullpath, 'a+', newline='') as write_obj:
    csv_writer = csv.writer(write_obj)
    csv_writer.writerow([candi_count, dp_count, blksum_count, inpsum_count])