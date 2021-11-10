import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import numpy as np
import cvxpy as cp
import time
from copy import deepcopy
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor,as_completed,wait

def lpsolve(vars,cons,obj,solver=cp.GUROBI):
    # The lp solver? Very interesting that input domain "vars" is not used
    prob=cp.Problem(obj,cons)
    prob.solve(solver=solver)
    return prob.value

class neuron(object):
    """
    Attributes:
        algebra_lower (numpy ndarray of float): neuron's algebra lower bound(coeffients of previous neurons and a constant) # the symbolic constraints
        algebra_upper (numpy ndarray of float): neuron's algebra upper bound(coeffients of previous neurons and a constant)
        concrete_algebra_lower (numpy ndarray of float): neuron's algebra lower bound(coeffients of input neurons and a constant) # the back-substituted constraint over input
        concrete_algebra_upper (numpy ndarray of float): neuron's algebra upper bound(coeffients of input neurons and a constant)
        concrete_lower (float): neuron's concrete lower bound
        concrete_upper (float): neuron's concrete upper bound
        concrete_highest_lower (float): neuron's highest concrete lower bound # the most precise bound
        concrete_lowest_upper (float): neuron's lowest concrete upper bound
        weight (numpy ndarray of float): neuron's weight # the direct connection
        bias (float): neuron's bias
        certain_flag (int): 0 uncertain 1 activated(>=0) 2 deactivated(<=0) # the status of relu node
        prev_abs_mode (int): indicates abstract mode of relu nodes in previous iteration.0 use first,1 use second # use this domain to check is relu status changes from uncertain to certain
    """

    def __init__(self):
        # initialization of one neuron
        self.algebra_lower=None
        self.algebra_upper=None
        self.concrete_algebra_lower=None
        self.concrete_algebra_upper=None
        self.concrete_lower=None
        self.concrete_upper=None
        self.concrete_highest_lower=None
        self.concrete_lowest_upper=None
        self.weight=None
        self.bias=None
        self.prev_abs_mode=None
        self.certain_flag=0 # relu status is uncertain at the beginning
    
    def clear(self):
        # clear the concrete bounds and relu status flags
        self.certain_flag=0
        self.concrete_highest_lower=None
        self.concrete_lowest_upper=None
        self.prev_abs_mode=None

    def print(self):
        # print out the neuron's information
        print('algebra_lower:',self.algebra_lower)
        print('algebra_upper:',self.algebra_upper)
        print('concrete_algebra_lower:',self.concrete_algebra_lower)
        print('concrete_algebra_upper:',self.concrete_algebra_upper)
        print('concrete_lower:',self.concrete_lower)
        print('concrete_upper:',self.concrete_upper)
        print('weight:',self.weight)
        print('bias:',self.bias)
        print('certain_flag:',self.certain_flag)

class layer(object):
    """
    Attributes:
        neurons (list of neuron): Layer neurons
        size (int): Layer size # number of neurons in one layer
        layer_type (int) : Layer type 0 input 1 affine 2 relu # layer type indicator
    """
    INPUT_LAYER=0
    AFFINE_LAYER=1
    RELU_LAYER=2    

    def __init__(self):
        self.size=None
        self.neurons=None
        self.layer_type=None
    
    def clear(self):
        # clear each neuron
        for i in range(len(self.neurons)):
            self.neurons[i].clear()

    def print(self):
        # print all the info
        print('Layer size:',self.size)
        print('Layer type:',self.layer_type)
        print('Neurons:')
        for neu in self.neurons:
            neu.print()
            print('\n')

class network(object):
    
    """
    Attributes:
        numLayers (int): Number of weight matrices or bias vectors in neural network
        layerSizes (list of ints): Size of input layer, hidden layers, and output layer
        inputSize (int): Size of input
        outputSize (int): Size of output
        mins (list of floats): Minimum values of inputs
        maxes (list of floats): Maximum values of inputs
        means (list of floats): Means of inputs and mean of outputs # means??why need means???
        ranges (list of floats): Ranges of inputs and range of outputs
        layers (list of layer): Network Layers
        unsafe_region (list of ndarray):coeffient of output and a constant
        property_flag (bool) : indicates the network have verification layer or not
        property_region (float) : Area of the input box # the sub input region??
        abs_mode_changed (int) : count of uncertain relu abstract mode changed # the number of refined relu nodes
        self.MODE_ROBUSTNESS=1
        self.MODE_QUANTITIVE=0
    """

    def __init__(self):
        self.MODE_QUANTITIVE=0
        self.MODE_ROBUSTNESS=1

        self.numlayers=None
        self.layerSizes=None
        self.inputSize=None
        self.outputSize=None
        self.mins=None
        self.maxes=None
        self.ranges=None
        self.layers=None
        self.property_flag=None
        self.property_region=None
        self.abs_mode_changed=None
    
    def clear(self):
        for i in range(len(self.layers)):
            self.layers[i].clear()
            # the clear is just clear the concrete bounds information
    
    def verify_lp_split(self,PROPERTY,DELTA,MAX_ITER=5,SPLIT_NUM=0,WORKERS=12,TRIM=False,SOLVER=cp.GUROBI,MODE=0,USE_OPT_2=False):
        # execution and result of DeepSRGR, SPLIT_NUM=0 for normal robustness verification
        # SPLIT_NUM is only used for quantitative for ACAS Xu
        if SPLIT_NUM>self.inputSize:
            SPLIT_NUM=self.inputSize
        if self.property_flag==True:
            # In previous deeppoly execution, pop out the last auxiliary layer and reload robustness property to be verified
            self.layers.pop()
            self.property_flag=False
        # reload the robustness property to be verified.
        self.load_robustness(PROPERTY,DELTA,TRIM=TRIM)
        # delta_list records all the input intervals
        delta_list=[self.layers[0].neurons[i].concrete_upper-self.layers[0].neurons[i].concrete_lower for i in range(self.inputSize)]
        self.clear()
        # Hmmmm, run deeppoly again...
        self.deeppoly()
        verify_layer=self.layers[-1]
        # verify_layer is the auxiliary layer
        verify_neuron_upper=np.array([neur.concrete_upper for neur in verify_layer.neurons])
        # take the upper bound of auxiliary neurons and sort them
        verify_list=np.argsort(verify_neuron_upper)
        # np.argsort() - Returns the indices that would sort an array in ascending order
        for i in range(len(verify_list)):
            if verify_neuron_upper[verify_list[i]]>=0:
                verify_list=verify_list[i:]
                break
        if verify_neuron_upper[verify_list[0]]<0:
            # will enter this branch if all the upper bounds < 0
            print("Property Verified")
            if MODE==self.MODE_ROBUSTNESS:
                return True
            return

        # generate the split sub-input spaces for quantitative verification
        split_list=[]
        for i in range(2**SPLIT_NUM):
            cur_split=[]
            for j in range(SPLIT_NUM):
                if i & (2**j) == 0:
                    cur_split.append([self.layers[0].neurons[j].concrete_lower,(self.layers[0].neurons[j].concrete_upper+self.layers[0].neurons[j].concrete_lower)/2])
                else:
                    cur_split.append([(self.layers[0].neurons[j].concrete_upper+self.layers[0].neurons[j].concrete_lower)/2,self.layers[0].neurons[j].concrete_upper])
            for j in range(SPLIT_NUM,self.inputSize):
                cur_split.append([self.layers[0].neurons[j].concrete_lower,self.layers[0].neurons[j].concrete_upper])
            split_list.append(cur_split)
        
        obj=None
        prob=None
        constraints=None
        variables=[]
        for i in range(len(self.layers)):
            # variable with shape=layer size
            variables.append(cp.Variable(self.layers[i].size))
        unsafe_set=set()
        unsafe_set_deeppoly=set()
        unsafe_area_list=np.zeros(len(split_list))
        verified_list=[]
        verified_area=0
        for i in verify_list:
            # the auxiliary neuron index, and after pruning, verify_list only leaves with those auxiliary neuron with violated concrete upper bound
            verification_neuron=self.layers[-1].neurons[i]
            # take the violated auxiliary neuron
            total_area=0
            for splits_num in range(len(split_list)):
                splits=split_list[splits_num]
                # operate over the sub-input space, where in robustness verification, this splits is the whole input space
                assert(len(splits)==self.inputSize)
                for j in range(self.inputSize):
                    # re-assign the input space
                    self.layers[0].neurons[j].concrete_lower=splits[j][0]
                    self.layers[0].neurons[j].concrete_algebra_lower=np.array([splits[j][0]])
                    self.layers[0].neurons[j].algebra_lower=np.array([splits[j][0]])
                    # all the bounds or constraints of input neuron will be the concrete interval
                    self.layers[0].neurons[j].concrete_upper=splits[j][1]
                    self.layers[0].neurons[j].concrete_algebra_upper=np.array([splits[j][1]])
                    self.layers[0].neurons[j].algebra_upper=np.array([splits[j][1]])            
                self.clear()
                # clear all the concrete bounds/relu status flag/relu abstract mode
                for j in range(MAX_ITER):
                    # Until reach the maximum number of iterations
                    self.deeppoly()
                    # run deeppoly again with the new input space, self.abs_mode_changed will be reset to 0 upon calling to deeppoly
                    # and accumulate once if the abstract mode of relu node is changed
                    # clear() function set prev_abs_mode in neuron to be None, with None assignment, self.abs_mode_changed will not accumulate
                    print("Abstract Mode Changed:",self.abs_mode_changed)
                    if (j==0) and (verification_neuron.concrete_upper>0):
                        # first refinement iteration for this verification_neuron, this is the current result on sub-input space already
                        # enter this branch meaning that, directly run dp on this sub-input space still cannot prune this violated aux neuron
                        unsafe_set_deeppoly.add(splits_num)
                        # The add() method doesn't add an element to the set if it's already present in it otherwise it
                        # meaning that under deeppoly, this sub-region is a unsafe region
                    constraints=[]
                    #Build Constraints for each layer
                    for k in range(len(self.layers)):
                        cur_layer=self.layers[k]
                        cur_neuron_list=cur_layer.neurons
                        if cur_layer.layer_type==layer.INPUT_LAYER:
                            for p in range(cur_layer.size):
                                # the sub-input space recorded in splits
                                constraints.append(variables[k][p]>=cur_neuron_list[p].concrete_lower)
                                constraints.append(variables[k][p]<=cur_neuron_list[p].concrete_upper)
                        elif cur_layer.layer_type==layer.AFFINE_LAYER:
                            assert(k>0)
                            for p in range(cur_layer.size):
                                constraints.append(variables[k][p]==cur_neuron_list[p].weight@variables[k-1]+cur_neuron_list[p].bias)
                                # the affine equality constraints
                        elif cur_layer.layer_type==layer.RELU_LAYER:
                            assert(cur_layer.size==self.layers[k-1].size)
                            assert(k>0)
                            for p in range(cur_layer.size):
                                constraints.append(variables[k][p]<=cur_neuron_list[p].algebra_upper[:-1]@variables[k-1]+cur_neuron_list[p].algebra_upper[-1])
                                # the upper constraints, containing the weights and bias
                                # constraints.append(variables[k][p]>=cur_neuron_list[p].algebra_lower[:-1]@variables[k-1]+cur_neuron_list[p].algebra_lower[-1])
                                # Modified:using two lower bounds
                                constraints.append(variables[k][p]>=0)
                                constraints.append(variables[k][p]>=variables[k-1][p])
                    #Build the verification neuron constraint
                    for k in verified_list:
                        # for previously verified list, add them into the constraints also to reduce the solved space
                        constraints.append(variables[-1][k]<=0)
                    #Modified:If MODE IS ROBUSTNESS AND USE_OPT_2 IS TRUE THEN CONSTRAINTS CAN BE ==0
                    if MODE==self.MODE_ROBUSTNESS and USE_OPT_2==True:
                        # this will lead to a smaller solved space
                        constraints.append(variables[-1][i]==0)
                    else:
                        constraints.append(variables[-1][i]>=0)
                    
                    #Simply check the feasibility, without objective function
                    prob=cp.Problem(cp.Maximize(0),constraints)
                    prob.solve(solver=SOLVER)
                    if prob.status!=cp.OPTIMAL:                        
                        print("Split:",splits_num,"Infeasible")
                        # break out of the MAX_ITERATION loop
                        break

                    #Refresh the input layer bounds
                    mppool=mp.Pool(WORKERS)
                    # python multiprocessing, the Pool object which offers a convenient means of parallelizing the execution of a function across multiple input values
                    tasklist=[]
                    input_neurons=self.layers[0].neurons
                    for k in range(self.inputSize):
                        obj=cp.Minimize(variables[0][k])
                        #Below using mp Pool
                        tasklist.append((variables,constraints,obj,SOLVER))
                        # tasklist takes in the parameters sent to lpsolve() function
                        obj=cp.Maximize(variables[0][k])
                        #Below using mp Pool
                        tasklist.append((variables,constraints,obj,SOLVER))
                    #Below using mp Pool
                    resultlist=mppool.starmap(lpsolve,tasklist)
                    # starmap() terminates when the result is ready
                    mppool.terminate()
                    # close this pool
                    for k in range(self.inputSize):
                        if resultlist[k*2]>=input_neurons[k].concrete_lower:
                            # if solve out a more precise lower bound, do the replacement for the input space
                            input_neurons[k].concrete_lower=resultlist[k*2]
                            input_neurons[k].concrete_algebra_lower=np.array([resultlist[k*2]])
                            input_neurons[k].algebra_lower=np.array([resultlist[k*2]])
                        #
                        if resultlist[k*2+1]<=input_neurons[k].concrete_upper:
                            # if solve out a more precise upper bound, do the replacement for the input space
                            input_neurons[k].concrete_upper=resultlist[k*2+1]
                            input_neurons[k].concrete_algebra_upper=np.array([resultlist[k*2+1]])
                            input_neurons[k].algebra_upper=np.array([resultlist[k*2+1]])

                    #Refresh the uncertain ReLu's lowerbound
                    mppool=mp.Pool(WORKERS)
                    count=0
                    # count for how many relu nodes now becomes certain
                    tasklist=[]
                    for k in range(len(self.layers)-1):
                        cur_layer=self.layers[k]
                        next_layer=self.layers[k+1]
                        if cur_layer.layer_type==layer.AFFINE_LAYER and next_layer.layer_type==layer.RELU_LAYER:
                            assert(cur_layer.size==next_layer.size)
                            for p in range(cur_layer.size):
                                if next_layer.neurons[p].certain_flag==0:
                                    # 0 flag means uncertain
                                    obj=cp.Minimize(variables[k][p])
                                    # to refine the input of relu layer, so this variables[k][p] is actually an affine neuron
                                    #Below using mp Pool
                                    tasklist.append((variables,constraints,obj,SOLVER))
                    # Below using mp Pool
                    resultlist=mppool.starmap(lpsolve,tasklist)
                    mppool.terminate()
                    index=0
                    for k in range(len(self.layers)-1):
                        cur_layer=self.layers[k]
                        next_layer=self.layers[k+1]
                        if cur_layer.layer_type==layer.AFFINE_LAYER and next_layer.layer_type==layer.RELU_LAYER:
                            assert(cur_layer.size==next_layer.size)
                            for p in range(cur_layer.size):
                                if next_layer.neurons[p].certain_flag==0:
                                    # the same conditions to filter out previous uncertain relu nodes
                                    if resultlist[index]>cur_layer.neurons[p].concrete_highest_lower:
                                        # this correct layer is affine layer
                                        cur_layer.neurons[p].concrete_highest_lower=resultlist[index]
                                    if resultlist[index]>=0:
                                        next_layer.neurons[p].certain_flag=1
                                        count+=1
                                    index+=1

                    #Refresh the uncertain ReLu's upperbound
                    mppool=mp.Pool(WORKERS)
                    tasklist=[]
                    for k in range(len(self.layers)-1):
                        cur_layer=self.layers[k]
                        next_layer=self.layers[k+1]
                        if cur_layer.layer_type==layer.AFFINE_LAYER and next_layer.layer_type==layer.RELU_LAYER:
                            assert(cur_layer.size==next_layer.size)
                            for p in range(cur_layer.size):
                                if next_layer.neurons[p].certain_flag==0:
                                    obj=cp.Maximize(variables[k][p])
                                    #Below using mp Pool
                                    tasklist.append((variables,constraints,obj,SOLVER))
                    # Below using mp Pool
                    resultlist=mppool.starmap(lpsolve,tasklist)
                    mppool.terminate()
                    index=0
                    for k in range(len(self.layers)-1):
                        cur_layer=self.layers[k]
                        next_layer=self.layers[k+1]
                        if cur_layer.layer_type==layer.AFFINE_LAYER and next_layer.layer_type==layer.RELU_LAYER:
                            assert(cur_layer.size==next_layer.size)
                            for p in range(cur_layer.size):
                                if next_layer.neurons[p].certain_flag==0:
                                    if resultlist[index]<cur_layer.neurons[p].concrete_lowest_upper:
                                        cur_layer.neurons[p].concrete_lowest_upper=resultlist[index]
                                        # if you set up concrete_lowest_upper in this way, in the next iteration of deeppoly, if the newly computed candidate bound is
                                        # less precise than this, this bound will not disappear!
                                    if resultlist[index]<=0:
                                        next_layer.neurons[p].certain_flag=2
                                        count+=1
                                    index+=1
                    print('Refreshed ReLu nodes:',count)

                if prob.status==cp.OPTIMAL:
                    # if after the MAX_ITER, still fail to verify the region to be spurious, then verification failed for this sub-region
                    area=1
                    for j in range(self.inputSize):
                        area*=(self.layers[0].neurons[j].concrete_upper-self.layers[0].neurons[j].concrete_lower)/delta_list[j]
                    # estimate the sub-area of this split that fails the verification
                    print("Split:",splits_num,"Area:",area*100)
                    if area>0:
                        if MODE==self.MODE_ROBUSTNESS:
                            # completely terminate the algorithm with this conclusion
                            return False
                        unsafe_area_list[splits_num]+=area
                        unsafe_set.add(splits_num)
                        total_area+=area # total unsafe area
            print('verification neuron:',i,'Unsafe Overapproximate(Box)%:',total_area*100)
            verified_area+=total_area
            verified_list.append(i)
        print('Overall Unsafe Overapproximate(Area)%',verified_area*100)
        verified_area=0
        for i in unsafe_area_list:
            if i>1/len(unsafe_area_list):
                # if the unsafe_area_list in a split overpass the whole split area
                verified_area+=1/len(unsafe_area_list)
            else:
                verified_area+=i
                # this verified_area is actually the area that fails the verification
        print('Overall Unsafe Overapproximate(Smart Area)%',verified_area*100)
        # the following two is counted over the number of unsafe splits over total number
        print('Overall Unsafe Overapproximate(Box)%:',len(unsafe_set)/len(split_list)*100)
        print('Overall Unsafe Overapproximate(Deeppoly)%:',len(unsafe_set_deeppoly)/len(split_list)*100)
        if MODE==self.MODE_ROBUSTNESS:
            # if any failure of the previous verification, will return False in line 386, so after enter here, is already verified
            return True
        if MODE==self.MODE_QUANTITIVE:
            if verified_area<len(unsafe_set)/len(split_list): # two types of way to estimate unsafe area
                # return the smaller area estimation
                return [verified_area*100,len(unsafe_set_deeppoly)/len(split_list)*100]
            else:
                return [len(unsafe_set)/len(split_list)*100,len(unsafe_set_deeppoly)/len(split_list)*100]

    def deeppoly(self):

        def pre(cur_neuron,i):
            # I assume this process is doing back-substitution to get the constraint defined over input neurons
            if i==0:
                # i is the layer index, so this branch means cur_neuron is the first hidden layer
                cur_neuron.concrete_algebra_lower=deepcopy(cur_neuron.algebra_lower)
                # deepcopy() return a new object with the same content
                cur_neuron.concrete_algebra_upper=deepcopy(cur_neuron.algebra_upper)
                # therefore the over-input constraint is the original symbolic constraint that this neuron has
            lower_bound=deepcopy(cur_neuron.algebra_lower)
            upper_bound=deepcopy(cur_neuron.algebra_upper)
            # take two copies of the symbolic constraints
            for k in range(i+1)[::-1]: # the list in order [i, i-1, ..., 1, 0]
                # after ending the iteration with k = 1, the constraint is defined over input neurons already
                # for k = 0, you evaluate the input constraint with input interval to get the concrete bounds
                # print(k)
                tmp_lower=np.zeros(len(self.layers[k].neurons[0].algebra_lower))
                # the len within the bracket is, the number of neurons in layer k-1. All element is set to 0 for initialization
                tmp_upper=np.zeros(len(self.layers[k].neurons[0].algebra_lower))
                assert(self.layers[k].size+1==len(lower_bound))
                # Current lower_bound is defined over layer k, and contains an extra bias.
                assert(self.layers[k].size+1==len(upper_bound))
                for p in range(self.layers[k].size):
                    # for each neuron in layer k (which is the variable in current constraint, replace this variable with its symbolic constraints)
                    if lower_bound[p]>=0:  
                        # print(lower_bound[p]*self.layers[k].neurons[p].algebra_lower)
                        # if the coefficients for lower constraint is positive, then replace with symbolic lower bound
                        # temp_lower is a list contains all coefficients with bias, so this is list addition
                        tmp_lower+=lower_bound[p]*self.layers[k].neurons[p].algebra_lower
                    else:
                        # print(lower_bound[p]*self.layers[k].neurons[p].algebra_upper)
                        # if the coefficients for lower constraint is negative, then replace with symbolic lower bound
                        tmp_lower+=lower_bound[p]*self.layers[k].neurons[p].algebra_upper

                    if upper_bound[p]>=0:
                        # similar strategy for the upper constraint substitution
                        tmp_upper+=upper_bound[p]*self.layers[k].neurons[p].algebra_upper
                    else:
                        tmp_upper+=upper_bound[p]*self.layers[k].neurons[p].algebra_lower                
                # print(tmp_lower)
                tmp_lower[-1]+=lower_bound[-1]
                # adding on the bias you have in the symbolic constraint
                tmp_upper[-1]+=upper_bound[-1]
                lower_bound=deepcopy(tmp_lower)
                # Update lower_bound with this newly constructed constraint, which is defined over k-1
                upper_bound=deepcopy(tmp_upper)
                if k==1:
                    # the current lower_ and upper_ bounds are already defined over k-1, so if k = 1, it means that we already define over input layer
                    cur_neuron.concrete_algebra_upper=deepcopy(upper_bound)
                    cur_neuron.concrete_algebra_lower=deepcopy(lower_bound)
            assert(len(lower_bound)==1)
            # at this point, input constraint is evaluated to the candidate concrete bound
            assert(len(upper_bound)==1)
            cur_neuron.concrete_lower=lower_bound[0]
            cur_neuron.concrete_upper=upper_bound[0]
            #add lowest and uppest history, get the most precise result
            if (cur_neuron.concrete_highest_lower==None) or (cur_neuron.concrete_highest_lower<cur_neuron.concrete_lower):
                cur_neuron.concrete_highest_lower=cur_neuron.concrete_lower
            if (cur_neuron.concrete_lowest_upper==None) or (cur_neuron.concrete_lowest_upper>cur_neuron.concrete_upper):
                cur_neuron.concrete_lowest_upper=cur_neuron.concrete_upper            


        self.abs_mode_changed=0 # (int) : count of uncertain relu abstract mode changed
        for i in range(len(self.layers)-1): # suppose we have n layers + 1 auxiliary layer, this range list is [0, n-1]
            # i is the layer index
            # print('i=',i)
            pre_layer=self.layers[i]
            cur_layer=self.layers[i+1]
            # cur_layer can be up to index n, so the auxiliary layer is also included in the analysis
            pre_neuron_list=pre_layer.neurons
            cur_neuron_list=cur_layer.neurons
            if cur_layer.layer_type==layer.AFFINE_LAYER:
                for j in range(cur_layer.size):
                    cur_neuron=cur_neuron_list[j]
                    # assign the symbolic constraints
                    cur_neuron.algebra_lower=np.append(cur_neuron.weight,[cur_neuron.bias])
                    cur_neuron.algebra_upper=np.append(cur_neuron.weight,[cur_neuron.bias])
                    # do the back-substitution to compute the concrete bound
                    pre(cur_neuron,i)
            elif cur_layer.layer_type==layer.RELU_LAYER:
                for j in range(cur_layer.size):
                    cur_neuron=cur_neuron_list[j]
                    pre_neuron=pre_neuron_list[j]
                    # the relu status is decided by the preceding neuron
                    #modified using lowest and uppest bound
                    if pre_neuron.concrete_highest_lower>=0 or cur_neuron.certain_flag==1:
                        # the relu node is always activated
                        cur_neuron.algebra_lower=np.zeros(cur_layer.size+1)
                        cur_neuron.algebra_upper=np.zeros(cur_layer.size+1)
                        cur_neuron.algebra_lower[j]=1
                        cur_neuron.algebra_upper[j]=1
                        # the constraint defined over input layer is the same
                        cur_neuron.concrete_algebra_lower=deepcopy(pre_neuron.concrete_algebra_lower)
                        cur_neuron.concrete_algebra_upper=deepcopy(pre_neuron.concrete_algebra_upper)
                        # concrete bounds are the same
                        cur_neuron.concrete_lower=pre_neuron.concrete_lower
                        cur_neuron.concrete_upper=pre_neuron.concrete_upper
                        #added
                        cur_neuron.concrete_highest_lower=pre_neuron.concrete_highest_lower
                        cur_neuron.concrete_lowest_upper=pre_neuron.concrete_lowest_upper
                        cur_neuron.certain_flag=1
                        # certain_flag (int): 0 uncertain 1 activated(>=0) 2 deactivated(<=0)
                    elif pre_neuron.concrete_lowest_upper<=0 or cur_neuron.certain_flag==2:
                        # relu node is always deactivated
                        cur_neuron.algebra_lower=np.zeros(cur_layer.size+1)
                        cur_neuron.algebra_upper=np.zeros(cur_layer.size+1)                        
                        cur_neuron.concrete_algebra_lower=np.zeros(self.inputSize)
                        cur_neuron.concrete_algebra_upper=np.zeros(self.inputSize)
                        cur_neuron.concrete_lower=0
                        cur_neuron.concrete_upper=0
                        #added
                        cur_neuron.concrete_highest_lower=0
                        cur_neuron.concrete_lowest_upper=0
                        cur_neuron.certain_flag=2
                    elif pre_neuron.concrete_highest_lower+pre_neuron.concrete_lowest_upper<=0:
                        #Relu abs mode changed to uncertain, node has more negative interval
                        if (cur_neuron.prev_abs_mode!=None) and (cur_neuron.prev_abs_mode!=0):
                            self.abs_mode_changed+=1
                        cur_neuron.prev_abs_mode=0
                        # I assume this abs_mode means that negative interval is more
                        cur_neuron.algebra_lower=np.zeros(cur_layer.size+1)
                        aux=pre_neuron.concrete_lowest_upper/(pre_neuron.concrete_lowest_upper-pre_neuron.concrete_highest_lower)
                        cur_neuron.algebra_upper=np.zeros(cur_layer.size+1)
                        cur_neuron.algebra_upper[j]=aux
                        cur_neuron.algebra_upper[-1]=-aux*pre_neuron.concrete_highest_lower
                        # Oh my, they also do substitution for relu node
                        pre(cur_neuron,i)
                    else:
                        #Relu abs mode changed
                        if (cur_neuron.prev_abs_mode!=None) and (cur_neuron.prev_abs_mode!=1):
                            self.abs_mode_changed+=1
                        cur_neuron.prev_abs_mode=1

                        cur_neuron.algebra_lower=np.zeros(cur_layer.size+1)
                        cur_neuron.algebra_lower[j]=1
                        aux=pre_neuron.concrete_lowest_upper/(pre_neuron.concrete_lowest_upper-pre_neuron.concrete_highest_lower)
                        cur_neuron.algebra_upper=np.zeros(cur_layer.size+1)
                        cur_neuron.algebra_upper[j]=aux
                        cur_neuron.algebra_upper[-1]=-aux*pre_neuron.concrete_highest_lower
                        pre(cur_neuron,i)

    def print(self):
        print('numlayers:%d' % (self.numLayers))
        print('layerSizes:',self.layerSizes)        
        print('inputSize:%d' % (self.inputSize))
        print('outputSize:%d' % (self.outputSize))
        print('mins:',self.mins)        
        print('maxes:',self.maxes)        
        print('ranges:',self.ranges)
        print('Layers:')
        for l in self.layers:
            l.print()
            print('\n')
    
    def load_property(self, filename):
        self.property_flag=True
        self.property_region=1
        with open(filename) as f:
            for i in range(self.layerSizes[0]):
                line = f.readline()
                linedata=[float(x) for x in line.strip().split(' ')]
                self.layers[0].neurons[i].concrete_lower=linedata[0]
                self.layers[0].neurons[i].concrete_upper=linedata[1]
                self.property_region*=linedata[1]-linedata[0]
                self.layers[0].neurons[i].concrete_algebra_lower=np.array([linedata[0]])
                self.layers[0].neurons[i].concrete_algebra_upper=np.array([linedata[1]])
                self.layers[0].neurons[i].algebra_lower=np.array([linedata[0]])
                self.layers[0].neurons[i].algebra_upper=np.array([linedata[1]])
                # print(linedata)
            self.unsafe_region=[]
            line=f.readline()
            verify_layer=layer()
            verify_layer.neurons=[]
            while line:                                
                linedata=[float(x) for x in line.strip().split(' ')]
                assert(len(linedata)==self.outputSize+1)
                verify_neuron=neuron()
                verify_neuron.weight=np.array(linedata[:-1])
                verify_neuron.bias=linedata[-1]
                verify_layer.neurons.append(verify_neuron)
                linedata=np.array(linedata)
                # print(linedata)
                self.unsafe_region.append(linedata)
                assert(len(linedata)==self.outputSize+1)
                line=f.readline()
            verify_layer.size=len(verify_layer.neurons)
            verify_layer.layer_type=layer.AFFINE_LAYER
            if len(verify_layer.neurons)>0:
                self.layers.append(verify_layer)

    def load_robustness(self, filename,delta, TRIM=False):
        # load the property they try to verify and use it to construct auxiliary layer
        if self.property_flag==True:
            # 'pop' returns the (i)th element of a list "lst". The element will be removed from the list as well
            self.layers.pop()
            # self.clear()
        self.property_flag=True        
        with open(filename) as f:
            self.property_region=1
            for i in range(self.layerSizes[0]):
                # the first part of lines are the image pixel intensities
                line = f.readline()
                # strip() removing both the leading and the trailing characters
                linedata=[float(line.strip())-delta,float(line.strip())+delta]
                # the pixel interval
                if TRIM:
                    # normalize the interval to be within [0,1]
                    if linedata[0]<0:
                        linedata[0]=0
                    if linedata[1]>1:
                        linedata[1]=1
                self.layers[0].neurons[i].concrete_lower=linedata[0]
                self.layers[0].neurons[i].concrete_upper=linedata[1]
                # the input interval is assigned to the input neurons
                self.property_region*=linedata[1]-linedata[0]
                # compute the total input area size
                self.layers[0].neurons[i].concrete_algebra_lower=np.array([linedata[0]])
                self.layers[0].neurons[i].concrete_algebra_upper=np.array([linedata[1]])
                self.layers[0].neurons[i].algebra_lower=np.array([linedata[0]])
                self.layers[0].neurons[i].algebra_upper=np.array([linedata[1]])
                # input neurons don't have symbolic constraints, jus set them simply as the concrete bounds
                # print(linedata)
            self.unsafe_region=[]
            line=f.readline()
            # declare the auxiliary layer, the robustness property is defined in the property file, in the form of Ax_b <= 0
            verify_layer=layer()
            verify_layer.neurons=[]
            while line:                                
                linedata=[float(x) for x in line.strip().split(' ')]
                # the coefficents and bias
                assert(len(linedata)==self.outputSize+1)
                verify_neuron=neuron()
                # set the weight for the auxiliary node
                verify_neuron.weight=np.array(linedata[:-1])
                # set the bias for the auxiliary node
                verify_neuron.bias=linedata[-1]
                verify_layer.neurons.append(verify_neuron)
                linedata=np.array(linedata)
                # print(linedata)
                self.unsafe_region.append(linedata)
                # unsafe_region adds in the auxiliary affine computation, what is the point of doing so??
                assert(len(linedata)==self.outputSize+1)
                line=f.readline()
            verify_layer.size=len(verify_layer.neurons)
            verify_layer.layer_type=layer.AFFINE_LAYER
            if len(verify_layer.neurons)>0:
                # add on the auxiliary layer to the self.layers list
                self.layers.append(verify_layer)
    
    def load_nnet(self, filename):
        with open(filename) as f:
            line = f.readline()
            cnt = 1
            while line[0:2] == "//":
                line=f.readline() 
                cnt+= 1
            #numLayers does't include the input layer!
            numLayers, inputSize, outputSize, _ = [int(x) for x in line.strip().split(",")[:-1]]
            line=f.readline()

            #input layer size, layer1size, layer2size...
            layerSizes = [int(x) for x in line.strip().split(",")[:-1]]

            line=f.readline()
            symmetric = int(line.strip().split(",")[0])

            line = f.readline()
            inputMinimums = [float(x) for x in line.strip().split(",")[:-1]]

            line = f.readline()
            inputMaximums = [float(x) for x in line.strip().split(",")[:-1]]

            line = f.readline()
            inputMeans = [float(x) for x in line.strip().split(",")[:-1]]

            line = f.readline()
            inputRanges = [float(x) for x in line.strip().split(",")[:-1]]

            #process the input layer
            self.layers=[]
            new_layer=layer()
            new_layer.layer_type=layer.INPUT_LAYER
            new_layer.size=layerSizes[0]
            new_layer.neurons=[]
            for i in range(layerSizes[0]):
                new_neuron=neuron()                
                new_layer.neurons.append(new_neuron)
            self.layers.append(new_layer)

            for layernum in range(numLayers):

                previousLayerSize = layerSizes[layernum]
                currentLayerSize = layerSizes[layernum+1]
                new_layer=layer()
                new_layer.size=currentLayerSize
                new_layer.layer_type=layer.AFFINE_LAYER
                new_layer.neurons=[]
                for i in range(currentLayerSize):
                    line=f.readline()
                    new_neuron=neuron()
                    aux = [float(x) for x in line.strip().split(",")[:-1]]
                    assert(len(aux)==previousLayerSize)
                    new_neuron.weight=np.array(aux)
                    new_layer.neurons.append(new_neuron)

                #biases                
                for i in range(currentLayerSize):
                    line=f.readline()
                    x = float(line.strip().split(",")[0])
                    new_layer.neurons[i].bias=x
                
                self.layers.append(new_layer)                

                #add relu layer
                if layernum+1==numLayers:
                    break
                new_layer=layer()
                new_layer.size=currentLayerSize
                new_layer.layer_type=layer.RELU_LAYER
                new_layer.neurons=[]
                for i in range(currentLayerSize):
                    new_neuron=neuron()
                    new_layer.neurons.append(new_neuron)
                self.layers.append(new_layer)

            self.numLayers = numLayers
            self.layerSizes = layerSizes
            self.inputSize = inputSize
            self.outputSize = outputSize
            self.mins = inputMinimums
            self.maxes = inputMaximums
            self.means = inputMeans
            self.ranges = inputRanges  
            self.property_flag=False        

    def load_rlv(self,filename):
        # the parser in DeepPoly, to collect the weights and bias information
        layersize=[]
        dicts=[]        
        self.layers=[]
        with open(filename,'r') as f:
            line=f.readline()
            while(line):            
                if(line.startswith('#')): # the new layer indicator
                    linedata=line.replace('\n','').split(' ')
                    # layersize records the list of layer size
                    layersize.append(int(linedata[3]))
                    layerdict={}
                    if(linedata[4]=='Input'):
                        # Input layer type
                        new_layer=layer()
                        new_layer.layer_type=layer.INPUT_LAYER
                        new_layer.size=layersize[-1]
                        new_layer.neurons=[]
                        for i in range(layersize[-1]):
                            # generate the corresponding number of neurons
                            new_neuron=neuron()
                            new_layer.neurons.append(new_neuron)
                            # reading from the next line
                            line=f.readline()
                            linedata=line.split(' ')
                            layerdict[linedata[1].replace('\n','')]=i
                            # in the dictionary, map 'inX0' to number 0, which is the corresponding index
                        dicts.append(layerdict)
                        self.layers.append(new_layer)
                    elif (linedata[4]=='ReLU'):
                        # Intermediate layer with ReLU activation function
                        new_layer=layer()
                        new_layer.layer_type=layer.AFFINE_LAYER # firstly encode the affine transformation
                        new_layer.size=layersize[-1]
                        new_layer.neurons=[]                        
                        for i in range(layersize[-1]):
                            new_neuron=neuron()
                            # initial the weight list, fully-connected therefore the number of coefficients is the layer size of preceding layer
                            new_neuron.weight=np.zeros(layersize[-2])
                            line=f.readline()
                            linedata=line.replace('\n','').split(' ')                        
                            layerdict[linedata[1]]=i
                            # again, map the neuron name string to the neuron index in this layer
                            new_neuron.bias=float(linedata[2])
                            # the bias is the first float number
                            nodeweight=linedata[3::2]
                            # take the coefficients from index 3 with stride = 2
                            nodename=linedata[4::2]
                            # take the variables from index 4 with stride = 2
                            assert(len(nodeweight)==len(nodename))
                            for j in range(len(nodeweight)):
                                new_neuron.weight[dicts[-1][nodename[j]]]=float(nodeweight[j])
                                # the coefficient is not displayed in index order, so re-arrange for the correct coefficent to be under correct index
                            new_layer.neurons.append(new_neuron)
                        self.layers.append(new_layer)                                     
                        dicts.append(layerdict)
                        #add relu layer
                        new_layer=layer()
                        new_layer.layer_type=layer.RELU_LAYER
                        new_layer.size=layersize[-1]
                        new_layer.neurons=[]
                        for i in range(layersize[-1]):
                            new_neuron=neuron()
                            new_layer.neurons.append(new_neuron)
                            # also declare new neuron object, but the information within is not filled in yet.
                        self.layers.append(new_layer)
                    elif (linedata[4]=='Linear') and (linedata[5]!='Accuracy'):
                        # eg: Linear res, the output layer, where the Accuracy is the auxiliary layer
                        new_layer=layer()
                        new_layer.layer_type=layer.AFFINE_LAYER
                        new_layer.size=layersize[-1]
                        new_layer.neurons=[]                        
                        for i in range(layersize[-1]):
                            new_neuron=neuron()
                            new_neuron.weight=np.zeros(layersize[-2])
                            line=f.readline()
                            linedata=line.replace('\n','').split(' ')                        
                            layerdict[linedata[1]]=i
                            new_neuron.bias=float(linedata[2])                            
                            nodeweight=linedata[3::2]
                            nodename=linedata[4::2]
                            assert(len(nodeweight)==len(nodename))
                            for j in range(len(nodeweight)):
                                new_neuron.weight[dicts[-1][nodename[j]]]=float(nodeweight[j])
                            new_layer.neurons.append(new_neuron)
                        self.layers.append(new_layer)                                     
                        dicts.append(layerdict)
                line=f.readline()
        self.layerSizes=layersize
        self.inputSize=layersize[0]
        self.outputSize=layersize[-1]
        self.numLayers=len(layersize)-1
        # the auxiliary layer is also counted
        pass

    def find_max_disturbance(self,PROPERTY,L=0,R=1000,TRIM=False):
        ans=0
        while L<=R:
            # print(L,R)
            mid=int((L+R)/2)
            self.load_robustness(PROPERTY,mid/1000,TRIM=TRIM)
            self.clear()
            self.deeppoly()
            flag=True
            for neuron_i in self.layers[-1].neurons:
                # print(neuron_i.concrete_upper)
                if neuron_i.concrete_upper>0:
                    flag=False
            if flag==True:
                ans=mid/1000
                L=mid+1
            else:
                R=mid-1
        return ans
    
    def find_max_disturbance_lp(self,PROPERTY,L,R,TRIM,WORKERS=12,SOLVER=cp.GUROBI):
        ans=L
        while L<=R:
            mid=int((L+R)/2)
            if self.verify_lp_split(PROPERTY=PROPERTY,DELTA=mid/1000,MAX_ITER=5,SPLIT_NUM=0,WORKERS=WORKERS,TRIM=TRIM,SOLVER=SOLVER,MODE=1):
                print("Disturbance:",mid/1000,"Success!")
                ans=mid/1000
                L=mid+1
            else:
                print("Disturbance:",mid/1000,"Failed!")
                R=mid-1
        return ans

def main():
    # # Experiment No.1
    # # Below shows Improvement in precision
    # # Notice: you can try different number for WORKER according to your working environment.
    # # Warning: To do all these experiments may be time consuming
    # # If you want to verify only a subset of this experiment, try MNIST_FNN_4 with mnist_0_local_property only.
    net_list=['rlv/caffeprototxt_AI2_MNIST_FNN_'+str(i)+'_testNetworkB.rlv' for i in range(2,3)] #range(2,9) for all experiments
    property_list=['properties/mnist_'+str(i)+'_local_property.in' for i in range(1)] #range(3) for all experiments
    rlist=[]
    for net_i in net_list:
        plist=[]
        for property_i in property_list:
            print("Net:",net_i,"Property:",property_i)
            net=network()
            net.load_rlv(net_i)
            delta_base=net.find_max_disturbance(PROPERTY=property_i,TRIM=True)            
            lp_ans=net.find_max_disturbance_lp(PROPERTY=property_i,L=int(delta_base*1000),R=int(delta_base*1000+63),TRIM=True,WORKERS=12,SOLVER=cp.CBC) #We use 96 in our experiment
            print("DeepPoly Max Verified Distrubance:",delta_base)
            print("Our method Max Verified Disturbance:",lp_ans)
            plist.append([delta_base,lp_ans])
        rlist.append(plist)
    print(rlist)

    # # Experiment No.2
    # # Below shows Robustness verification performance
    # # Notice: you can try different number for WORKER according to your working environment.
    # # Notice: you can try different net, property and delta. (FNN4, property0-49, 0.037), (FNN5, property0-49, 0.026), (FNN6, property0-49, 0.021), (FNN7, property0-49, 0.015) is used in paper.
    # # Notice: you can try different MAX_ITER to check how iteration numbers affect experiment results.
    # # Warning: To do batch verification in large nets is time consuming. Try FNN4 if you want to do quick reproducing.
    net_list=['rlv/caffeprototxt_AI2_MNIST_FNN_'+str(i)+'_testNetworkB.rlv' for i in range(4,5)] # You can try different nets.
    property_list=['properties/mnist_'+str(i)+'_local_property.in' for i in range(5)] # We use range(50) in our experiment.
    delta=0.037
    for net_i in net_list:
        pass_list=[]
        nopass_list=[]
        net=network()
        net.load_rlv(net_i)
        index=0
        count_deeppoly=0
        count_deepsrgr=0
        mintime=None
        maxtime=None
        avgtime=0
        print('Verifying Network:',net_i)
        for property_i in property_list:
            net.clear()
            net.load_robustness(property_i,delta,TRIM=True)
            net.deeppoly()
            flag=True
            for neuron_i in net.layers[-1].neurons:
                # print(neuron_i.concrete_upper)
                if neuron_i.concrete_upper>0:
                    flag=False
            if flag==True:
                count_deeppoly+=1
                print(property_i,'DeepPoly Success!')
            else:
                print(property_i,'DeepPoly Failed!')

            start=time.time()
            if net.verify_lp_split(PROPERTY=property_i,DELTA=delta,MAX_ITER=5,SPLIT_NUM=0,WORKERS=12,TRIM=True,SOLVER=cp.CBC,MODE=1,USE_OPT_2=True): #We use 96 WORKERS in our experiment
                print(property_i,'DeepSRGR Success!')
                count_deepsrgr+=1
                pass_list.append(index)
            else:
                print(property_i,'DeepSRGR Failed!')
                nopass_list.append(index)
            end=time.time()
            runningtime=end-start
            print(property_i,'Running Time',runningtime)
            if (mintime==None) or (runningtime<mintime):
                mintime=runningtime
            if (maxtime==None) or (runningtime>maxtime):
                maxtime=runningtime
            avgtime+=runningtime
            index+=1
        print('DeepPoly Verified:',count_deeppoly,'DeepSRGR Verified:',count_deepsrgr)
        print('Min Time:',mintime,'Max Time',maxtime,'Avg Time',avgtime/50)
        print('Passlist:',pass_list)
        print('Nopasslist:',nopass_list)    

    # # Experiment No.3
    # # Below is the Quantitive Robustness experiment
    # # Notice: you can try different number for WORKER according to your working environment.
    # # Warning: To do all these experiments may be time consuming
    # # If you want to verify only a subset of this experiment, try 4_2 net with disturbance 0.02 and local_robustness_2 only.
    # net_list=['nnet/ACASXU_experimental_v2a_4_2.nnet','nnet/ACASXU_experimental_v2a_4_3.nnet','nnet/ACASXU_experimental_v2a_4_4.nnet']
    net_list=['nnet/ACASXU_experimental_v2a_4_2.nnet']
    # property_list=['properties/local_robustness_2.txt','properties/local_robustness_3.txt','properties/local_robustness_4.txt','properties/local_robustness_5.txt','properties/local_robustness_6.txt']
    property_list=['properties/local_robustness_2.txt']
    # disturbance_list=[0.02,0.03,0.04,0.05,0.06]    
    disturbance_list=[0.02]
    rlist=[]
    for net_i in net_list:
        plist=[]
        for property_i in property_list:
            net=network()
            net.load_nnet(net_i)
            delta_base=net.find_max_disturbance(PROPERTY=property_i)
            dlist=[]
            for disturbance_i in disturbance_list:
                print("Net:",net_i,"Property:",property_i,"Delta:",delta_base+disturbance_i)
                start=time.time()
                net=network()
                net.load_nnet(net_i)
                dlist.append(net.verify_lp_split(PROPERTY=property_i,DELTA=delta_base+disturbance_i,MAX_ITER=5,WORKERS=12,SPLIT_NUM=5,SOLVER=cp.CBC)) #We use 96 WORKERS in our experiment
                end=time.time()
                print("Finished Time:",end-start)
            plist.append(dlist)
        rlist.append(plist)
    print(rlist)

    # # Below shows DeepPoly uncertain relus of Experiment No.1
    net_list=['rlv/caffeprototxt_AI2_MNIST_FNN_'+str(i)+'_testNetworkB.rlv' for i in range(2,9)] #8
    property_list=['properties/mnist_'+str(i)+'_local_property.in' for i in range(3)] #3
    radius_list=[[[0.034, 0.047], [0.017, 0.023], [0.017, 0.023]], [[0.049, 0.066], [0.025, 0.033], [0.045, 0.058]], [[0.045, 0.06], [0.024, 0.03], [0.035, 0.046]], [[0.034, 0.042], [0.016, 0.019], [0.021, 0.027]], [[0.022, 0.026], [0.011, 0.013], [0.021, 0.025]], [[0.021, 0.023], [0.01, 0.011], [0.017, 0.019]], [[0.037, 0.044], [0.02, 0.022], [0.033, 0.04]]]    
    net_index=0
    for net_i in net_list:
        print("Currnet Network:",net_i)
        property_index=0
        for property_i in property_list:
            print("Current Property",property_i)
            print("Current Radius",radius_list[net_index][property_index][1])
            net=network()
            net.load_rlv(net_i)
            net.load_robustness(property_i,radius_list[net_index][property_index][1],TRIM=True)
            net.deeppoly()
            count=0
            for layer_i in net.layers:
                if layer_i.layer_type==layer.RELU_LAYER:
                    for neuron_i in layer_i.neurons:
                        if neuron_i.certain_flag==0:
                            count+=1
            print("Uncertain Relu:",count)
            property_index+=1
        net_index+=1
    
    # # Below shows DeepPoly uncertain relus of Experiment No.2 (FNN_4 with only not pass property)
    net_list=['rlv/caffeprototxt_AI2_MNIST_FNN_'+str(i)+'_testNetworkB.rlv' for i in range(4,5)]
    property_list=['properties/mnist_'+str(i)+'_local_property.in' for i in [1, 8, 9, 12, 15, 17, 19, 21, 23, 24, 25, 35, 36, 38, 47]]
    radius=0.037    
    for net_i in net_list:
        print("Currnet Network:",net_i)        
        for property_i in property_list:
            print("Current Property",property_i)
            print("Current Radius",radius)
            net=network()
            net.load_rlv(net_i)
            net.load_robustness(property_i,radius,TRIM=True)
            net.deeppoly()
            count=0
            for layer_i in net.layers:
                if layer_i.layer_type==layer.RELU_LAYER:
                    for neuron_i in layer_i.neurons:
                        if neuron_i.certain_flag==0:
                            count+=1
            print("Uncertain Relu:",count)                    

    # # Below shows the small illustration example in paper
    # net=network()
    # net.load_rlv('rlv/smallexample.rlv')
    # net.verify_lp_split(PROPERTY='properties/smallexample.in',DELTA=1,MAX_ITER=2,SPLIT_NUM=0,WORKERS=12,SOLVER=cp.CBC,MODE=1,USE_OPT_2=False)
    # net.print()
    # net.load_robustness('properties/smallexample.in',1)
    # net.deeppoly()
    # net.print()
    # flag=True
    # for neuron_i in net.layers[-1].neurons:
    #     print(neuron_i.concrete_upper)
    #     if neuron_i.concrete_upper>0:
    #         flag=False
    # print(flag)

    pass
    

if __name__ == "__main__":
    main()