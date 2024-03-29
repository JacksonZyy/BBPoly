'''
@author: Adrian Hoffmann
'''

from elina_abstract0 import *
from elina_manager import *
from deeppoly_nodes import *
from functools import reduce
from ai_milp import milp_callback
import gc

class layers:
    def __init__(self):
        self.layertypes = []
        self.weights = []
        self.biases = []
        self.filters = []
        self.numfilters = []
        self.filter_size = [] 
        self.input_shape = []
        self.strides = []
        self.padding = []
        self.out_shapes = []
        self.pool_size = []
        self.numlayer = 0
        self.ffn_counter = 0
        self.conv_counter = 0
        self.residual_counter = 0
        self.pool_counter = 0
        self.activation_counter = 0
        self.specLB = []
        self.specUB = []
        self.original = []
        self.zonotope = []
        self.predecessors = []
        self.lastlayer = None
        self.last_weights = None
        self.label = -1
        self.prop = -1

    def calc_layerno(self):
        return self.ffn_counter + self.conv_counter + self.residual_counter + self.pool_counter + self.activation_counter

    def is_ffn(self):
        return not any(x in ['Conv2D', 'Conv2DNoReLU', 'Resadd', 'Resaddnorelu'] for x in self.layertypes)

    def set_last_weights(self, constraints):
        length = 0.0       
        last_weights = [0 for weights in self.weights[-1][0]]
        for or_list in constraints:
            for (i, j, cons) in or_list:
                if j == -1:
                    last_weights = [l + w_i + float(cons) for l,w_i in zip(last_weights, self.weights[-1][i])]
                else:
                    last_weights = [l + w_i + w_j + float(cons) for l,w_i, w_j in zip(last_weights, self.weights[-1][i], self.weights[-1][j])]
                length += 1
        self.last_weights = [w/length for w in last_weights]


    def back_propagate_gradiant(self, nlb, nub):
        #assert self.is_ffn(), 'only supported for FFN'

        grad_lower = self.last_weights.copy()
        grad_upper = self.last_weights.copy()
        last_layer_size = len(grad_lower)
        for layer in range(len(self.weights)-2, -1, -1):
            weights = self.weights[layer]
            lb = nlb[layer]
            ub = nub[layer]
            layer_size = len(weights[0])
            grad_l = [0] * layer_size
            grad_u = [0] * layer_size

            for j in range(last_layer_size):

                if ub[j] <= 0:
                    grad_lower[j], grad_upper[j] = 0, 0

                elif lb[j] <= 0:
                    grad_upper[j] = grad_upper[j] if grad_upper[j] > 0 else 0
                    grad_lower[j] = grad_lower[j] if grad_lower[j] < 0 else 0

                for i in range(layer_size):
                    if weights[j][i] >= 0:
                        grad_l[i] += weights[j][i] * grad_lower[j]
                        grad_u[i] += weights[j][i] * grad_upper[j]
                    else:
                        grad_l[i] += weights[j][i] * grad_upper[j]
                        grad_u[i] += weights[j][i] * grad_lower[j]
            last_layer_size = layer_size
            grad_lower = grad_l
            grad_upper = grad_u
        return grad_lower, grad_upper




class Analyzer:
    def __init__(self, ir_list, nn, domain, timeout_lp, timeout_milp, output_constraints, use_default_heuristic, label, prop, testing = False, layer_by_layer = False, is_residual = False, is_blk_segmentation=False, blk_size=0, is_early_terminate = False, early_termi_thre = 0, is_sum_def_over_input = True, var_cancel_heuristic = False):
        """
        Arguments
        ---------
        ir_list: list
            list of Node-Objects (e.g. from DeepzonoNodes), first one must create an abstract element
        domain: str
            either 'deepzono', 'refinezono' or 'deeppoly'
        """
        self.ir_list = ir_list
        self.is_greater = None
        self.man = None
        self.layer_by_layer = layer_by_layer
        self.is_residual = is_residual
        self.is_blk_segmentation = is_blk_segmentation
        self.blk_size = blk_size
        self.is_early_terminate = is_early_terminate
        self.early_termi_thre = early_termi_thre
        self.is_sum_def_over_input = is_sum_def_over_input
        self.var_cancel_heuristic = var_cancel_heuristic
        self.refine = False
        if domain == 'deeppoly' or domain == 'refinepoly':
            self.man = fppoly_manager_alloc()
            self.is_greater = is_greater
        self.domain = domain
        self.nn = nn
        self.timeout_lp = timeout_lp
        self.timeout_milp = timeout_milp
        self.output_constraints = output_constraints
        self.use_default_heuristic = use_default_heuristic
        self.testing = testing
        self.relu_groups = []
        self.label = label
        self.prop = prop
    
    def __del__(self):
        elina_manager_free(self.man)
        
    
    def get_abstract0(self):
        """
        processes self.ir_list and returns the resulting abstract element
        """
        element = self.ir_list[0].transformer(self.man)
        nlb = []
        nub = []
        testing_nlb = []
        testing_nub = []
        # print("The len of deeppolyNodes is ", len(self.ir_list))
        for i in range(1, len(self.ir_list)):
            #print(self.is_early_terminate)
            element_test_bounds = self.ir_list[i].transformer(self.nn, self.man, element, nlb, nub, self.relu_groups, 'refine' in self.domain, self.timeout_lp, self.timeout_milp, self.use_default_heuristic, self.testing, self.layer_by_layer, self.is_residual, self.is_blk_segmentation, self.blk_size, self.is_early_terminate, self.early_termi_thre, self.is_sum_def_over_input, self.var_cancel_heuristic)
            #print("Transformer done for ",i)
            if self.testing and isinstance(element_test_bounds, tuple):
                element, test_lb, test_ub = element_test_bounds
                testing_nlb.append(test_lb)
                testing_nub.append(test_ub)
            else:
                element = element_test_bounds
        if self.domain in ["refinezono", "refinepoly"]:
            gc.collect()
        if self.testing:
            return element, testing_nlb, testing_nub
        return element, nlb, nub
    
    
    def analyze(self):
        """
        analyses the network with the given input
        
        Returns
        -------
        output: int
            index of the dominant class. If no class dominates then returns -1
        """
        element, nlb, nub = self.get_abstract0()
        #print("leave get_abstract0()")
        output_size = 0
        output_size = self.ir_list[-1].output_length #reduce(lambda x,y: x*y, self.ir_list[-1].bias.shape, 1)
        dominant_class = -1
        label_failed = []
        x = None
        if self.output_constraints is None:
            candidate_labels = []
            if self.label == -1:
                for i in range(output_size):
                    candidate_labels.append(i)
            else:
                candidate_labels.append(self.label)
            adv_labels = []
            if self.prop == -1:
                for i in range(output_size):
                    adv_labels.append(i)
            else:
                adv_labels.append(self.prop)
            # print("adv_labels",adv_labels)   
            for i in candidate_labels:
                flag = True
                label = i
                for j in adv_labels:
                    if label!=j and not self.is_greater(self.man, element, label, j, self.use_default_heuristic, self.layer_by_layer, self.is_residual, self.is_blk_segmentation, self.blk_size, self.is_early_terminate, self.early_termi_thre, self.is_sum_def_over_input, self.var_cancel_heuristic):
                        # testing if label is always greater than j
                        flag = False
                        if self.label!=-1:
                            label_failed.append(j)
                        if config.complete == False:
                            break


                if flag:
                    dominant_class = i
                    break
        #print("enter abstract_free() in python")
        elina_abstract0_free(self.man, element)
        #print("End analyze() in python")
        return dominant_class, nlb, nub, label_failed, x
