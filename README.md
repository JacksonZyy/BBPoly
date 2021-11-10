BBPoly instantiated on ERAN
========

BBPoly is a scalabel and module robustness verification tool which is instantiated on DeepPoly domain in ETH Robustness Analyzer for Neural Networks ([ERAN](https://github.com/eth-sri/eran)). BBPoly provides incomplete verification of MNIST, CIFAR10 based networks can be tuned to achieve trade-off between precision and scalability (see recommended configuration settings at the bottom). 

BBPoly currently supports networks with ReLU activation and is sound under floating point arithmetic. It employs the abstract domain designed in DeepPoly and leverages network block summary to balance scalability and precision. The description of our BBPoly system can be found in [APLAS'21](https://link.springer.com/chapter/10.1007/978-3-030-89051-3_1) or [arXiv preprint](https://arxiv.org/abs/2108.11651)

For reference, ERAN is developed at the [SRI Lab, Department of Computer Science, ETH Zurich](https://www.sri.inf.ethz.ch/) as part of the [Safe AI project](http://safeai.ethz.ch/).


Requirements 
------------
GNU C compiler, ELINA, Gurobi's Python interface,

python3.6 or higher, tensorflow 1.11 or higher, numpy.


Installation
------------
Clone the BBPoly repository via git as follows:
```
git clone https://github.com/JacksonZyy/BBPoly.git
cd BBPoly
```

The dependencies for BBPoly can be installed step by step as follows (sudo rights might be required):

Install m4:
```
wget ftp://ftp.gnu.org/gnu/m4/m4-1.4.1.tar.gz
tar -xvzf m4-1.4.1.tar.gz
cd m4-1.4.1
./configure
make
make install
cp src/m4 /usr/bin
cd ..
rm m4-1.4.1.tar.gz
```

Install gmp:
```
wget https://gmplib.org/download/gmp/gmp-6.1.2.tar.xz
tar -xvf gmp-6.1.2.tar.xz
cd gmp-6.1.2
./configure --enable-cxx
make
make install
cd ..
rm gmp-6.1.2.tar.xz
```

Install mpfr:
```
wget https://www.mpfr.org/mpfr-current/mpfr-4.0.2.tar.xz
tar -xvf mpfr-4.0.2.tar.xz
cd mpfr-4.0.2
./configure
make
make install
cd ..
rm mpfr-4.0.2.tar.xz
```

Install cddlib:
```
wget https://github.com/cddlib/cddlib/releases/download/0.94j/cddlib-0.94j.tar.gz
tar -xvf cddlib-0.94j.tar.gz
cd cddlib-0.94j
./configure
make
make install
cd ..
rm cddlib-0.94j.tar.gz

```

Compile ELINA:
```
cd ELINA
./configure -use-deeppoly
make
make install
cd ..
```

Install Gurobi:
```
wget https://packages.gurobi.com/9.0/gurobi9.0.0_linux64.tar.gz
tar -xvf gurobi9.0.0_linux64.tar.gz
cd gurobi900/linux64/src/build
sed -ie 's/^C++FLAGS =.*$/& -fPIC/' Makefile
make
cp libgurobi_c++.a ../../lib/
cd ../../
cp lib/libgurobi90.so /usr/lib
python3 setup.py install
cd ../../

```

Update environment variables:
```
export GUROBI_HOME="Current_directory/gurobi900/linux64"
export PATH="${PATH}:${GUROBI_HOME}/bin"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib:${GUROBI_HOME}/lib

```

We also provide scripts that will compile ELINA and all the necessary dependencies. One can run it as follows:

```
sudo ./install.sh
source gurobi_setup_path.sh

```


Note that to run the system with Gurobi one needs to obtain an academic license for gurobi from https://user.gurobi.com/download/licenses/free-academic.

To install the remaining python dependencies (numpy and tensorflow), type:

```
pip3 install -r requirements.txt
```

BBPoly may not be compatible with older versions of tensorflow (according to ERAN system), so if you have an older version and want to keep it, then we recommend using the python virtual environment for installing tensorflow.


Usage
-------------

```
cd tf_verify

python3 . --netname <path to the network file> --epsilon <float between 0 and 1> --dataset <mnist/cifar10> [optional] --layer_by_layer <True/False> --is_residual <True/False> --is_blk_segmentation <True/False> --blk_size <int> --is_early_terminate <True/False> --early_termi_thre <int> --is_sum_def_over_input <True/False>
```

* ```<epsilon>```: specifies bound for the L∞-norm based perturbation (default is 0). This parameter is not required for testing_conv/fcn_main.py execution, since an epsilon list is already provided in these two files.

* ```<layer_by_layer>```: the flag indicating whether the back-substitution process will be conducted layer-by-layer or not. Should be used for abstract refinement only, the default value is false.

* ```<is_residual>```: whether the verification network is residual network or not (default is false).

* ```<is_blk_segmentation>```: specifies if the analysis will be conducted in a modular way, meaning that we will segment the network into blocks and leverage block summary to speed up the analysis process (default is false).

* ```<blk_size>```: indicates how many affine layers are contained in one block (default is 0). We use this parameter to segment the network, therefore it is only meaningful if ```<is_blk_segmentation>``` is activated

* ```<is_early_terminate>```: whether to terminate the back-substitution process earlier (default is false).

* ```<early_termi_thre>```: specifies the threhold of back-substitution steps, then we will terminate the back-substitution (default is 0). This parameter only makes sense if ```<is_early_terminate>``` is activated.

* ```<is_sum_def_over_input>```: specifies if the block summary is defined over the actual input layer of the network or not, since we have two types of summary, input summary or block summary (check [our paper](https://link.springer.com/chapter/10.1007/978-3-030-89051-3_1) for detail). The default value is false, so the default version of summary is block summary.

* We aim to conduct abstract refinement when the verification fails, to either prove the robustness or find a counterexample to falsify the robustness, w.r.t. the whole input space. If we fail to conclude in the two ways mentioned above, we will try to return quantitative result. Those are the future features to be added in the system. 



Example
-------------

L_oo Specification for BBPoly (block summmary) execution:
```
cd tf_verify
wget https://files.sri.inf.ethz.ch/eran/nets/tensorflow/mnist/mnist_relu_9_200.tf
python3 . --netname mnist_relu_9_200.tf --epsilon 0.005 --dataset mnist --blk_size 3 --is_blk_segmentation True 
```

will evaluate the local robustness of the MNIST fully-connected network with ReLU activation. We have 100 MNIST test images, where robustness is only verified for those images that are classifed correctly by the network. In the above setting, epsilon=0.005, we segment the network every time we accumulate three affine layers and the block type is default (block summary). Our analyzer will print the following:

* 'analysis precision', which is the ratio of images on which the network is robust versus the number of images on which it classifies correctly (also named as candidate images).

* 'average execution time', which is the average running time (in seconds) for candidate images.


L_oo Specification for BBPoly (input summmary) execution:
```
python3 . --netname mnist_relu_9_200.tf --epsilon 0.005 --dataset mnist --blk_size 3 --is_blk_segmentation True --is_sum_def_over_input True
```

will check robustness in a modular way, using input summary method.


L_oo Specification for DeepPoly execution:
```
python3 . --netname mnist_relu_9_200.tf --epsilon 0.005 --dataset mnist
```
is the default execution mode (DeepPoly), for details of DeepPoly, please refer to [POPL' 19](https://www.sri.inf.ethz.ch/publications/singh2019domain). 

Stress testing (fully-connected network) for DeepPoly vs BBPoly(block summmary) vs BBPoly(input summmary):

```
python3 testing_fcn_main.py --netname mnist_relu_9_200.tf --dataset mnist
```
will run three methods at the same time, with various epsilon in a pre-defined epsilon list. The verification result and execution time for each method/epsilon/image will be recorded in a csv file.


Stress testing (residual network) for BBPoly(block summmary) vs BBPoly(input summmary):

```
wget https://files.sri.inf.ethz.ch/eran/nets/onnx/cifar/ResNet18_DiffAI.onnx
python3 testing_res_main.py --netname ResNet18_DiffAI.onnx --dataset cifar10 --is_residual True
```
will verify robustness for the residual network, with a pre-defined epsilon to be 8/255 (according to [GPUPoly setup](https://www.sri.inf.ethz.ch/publications/mller2021neural)). DeepPoly fails to terminate for selected residual networks within 3 hours timeout, so we compare between different methods in BBPoly. The analysis block for residual network is the intrinsic residual block, and we compare between BlkSum_4bound and Input_Sum methods in BBPoly. BlkSum_4bound refers to block summary method with early termination in 4 steps of back-substitution; Input_Sum refers to input summary method. The verification result and execution time for each method/epsilon/image will be recorded in a csv file. (You can use ```<nohup>``` command if the execution takes long time.)


Publications
-------------
*  [Scalable and Modular Robustness Analysis of Deep Neural Networks](https://link.springer.com/chapter/10.1007/978-3-030-89051-3_1)

   Yuyi Zhong,  Quang-Trung Ta, Tianzuo Luo, Fanlong Zhang,  Siau-Cheng Khoo.
   
   APLAS 2021.


References
-------------
*  [An Abstract Domain for Certifying Neural Networks](https://www.sri.inf.ethz.ch/publications/singh2019domain).

    Gagandeep Singh, Timon Gehr, Markus Püschel, and Martin Vechev. 

    POPL 2019.

*  [Improving Neural Network Verification through Spurious Region Guided Refinement](https://link.springer.com/chapter/10.1007%2F978-3-030-72016-2_21).

    Pengfei Yang, Renjue Li, Jianlin Li, Cheng-Chao Huang, Jingyi Wang, Jun Sun, Bai Xue, and Lijun Zhang.

    TACAS 2021.



Neural Networks and Datasets
---------------

We collected a number of pretrained MNIST and CIAFR10 defended and undefended feedforward, convolutional and residual neural networks with ReLU activations, from [ERAN REPO](https://github.com/eth-sri/eran). The adversarial training to obtain the defended networks is performed using [DiffAI](https://github.com/eth-sri/diffai). 

| Dataset  |   Model  |  Type   | #units | #layers| Activation | Training Defense| Download |
| :-------- | :-------- | :-------- | :-------------| :-------------| :------------ | :------------- | :---------------:|
| MNIST   | 9x200 | fully connected | 1,610 | 9   | ReLU | None | [:arrow_down:](https://files.sri.inf.ethz.ch/eran/nets/tensorflow/mnist/mnist_relu_9_200.tf)|
|         | 6x500 | fully connected | 3,000 | 6   | ReLU | None | [:arrow_down:](https://files.sri.inf.ethz.ch/eran/nets/pytorch/mnist/ffnnRELU__Point_6_500.pyt)|
|         | ConvBig | convolutional | 48,064 | 6  | ReLU | DiffAI | [:arrow_down:](https://files.sri.inf.ethz.ch/eran/nets/pytorch/mnist/convBigRELU__DiffAI.pyt) |
|         | ConvSuper | convolutional | 88,544 | 6  | ReLU | DiffAI | [:arrow_down:](https://files.sri.inf.ethz.ch/eran/nets/pytorch/mnist/convSuperRELU__DiffAI.pyt) |
| CIFAR10 | 9x200 | fully connected | 1,810 | 9 | ReLU | None | [:arrow_down:](https://files.sri.inf.ethz.ch/eran/nets/tensorflow/cifar/cifar_relu_9_200.tf) |
|         | 6x500 | fully connected | 3,000 | 6   | ReLU | None | [:arrow_down:](https://files.sri.inf.ethz.ch/eran/nets/pytorch/cifar/ffnnRELU__Point_6_500.pyt)|
|         | ConvBig | convolutional | 62,464 | 6 | ReLU | DiffAI | [:arrow_down:](https://files.sri.inf.ethz.ch/eran/nets/pytorch/cifar/convBigRELU__DiffAI.pyt) | 
|         | ResNet18 | Residual | 558K | 19 | ReLU | DiffAI | [:arrow_down:](https://files.sri.inf.ethz.ch/eran/nets/onnx/cifar/ResNet18_DiffAI.onnx) |
|         | SkipNet18 | Residual | 558K | 19 | ReLU | DiffAI | [:arrow_down:](https://files.sri.inf.ethz.ch/eran/nets/onnx/cifar/SkipNet18_DiffAI.onnx) |
|         | ResNet34 | Residual | 967K | 35 | ReLU | DiffAI | [:arrow_down:](https://files.sri.inf.ethz.ch/eran/nets/onnx/cifar/ResNet34_DiffAI.onnx) |

The first 100 images from the testset of both MNIST and CIFAR10 datasets are provided in the 'data' folder. The analyzer first verifies whether the neural network classifies an image correctly before performing robustness analysis.

Experimental Results
--------------
We have implemented our proposed method in a prototype analyzer called BBPoly, which is built on top of DeepPoly. Then, we conducted extensive experiments to evaluate the performance of both [our tool](https://arxiv.org/abs/2108.11651) and [DeepPoly](https://www.sri.inf.ethz.ch/publications/singh2019domain), in terms of precision, memory usage and runtime. The evaluation machine is equipped
with a 2600 MHz 24 core GenuineIntel CPU with 64 GB of RAM.

![High Level](https://files.sri.inf.ethz.ch/eran/plots/mnist_6_100.png)

In the following, we compare the precision and performance of DeepZ and DeepPoly on a subset of the neural networks listed above in multi-threaded mode. In can be seen that DeepPoly is overall more precise than DeepZ but it is slower than DeepZ on the convolutional networks. 

![High Level](https://files.sri.inf.ethz.ch/eran/plots/mnist_6_500.png)

![High Level](https://files.sri.inf.ethz.ch/eran/plots/mnist_convsmall.png)

![High Level](https://files.sri.inf.ethz.ch/eran/plots/mnist_sigmoid_tanh.png)

![High Level](https://files.sri.inf.ethz.ch/eran/plots/cifar10_convsmall.png)


The table below compares the performance and precision of DeepZ and DeepPoly on our large networks trained with DiffAI. 


<table aligh="center">
  <tr>
    <td>Dataset</td>
    <td>Model</td>
    <td>&epsilon;</td>
    <td colspan="2">% Verified Robustness</td>
    <td colspan="2">% Average Runtime (s)</td>
  </tr>
  <tr>
   <td> </td>
   <td> </td>
   <td> </td>
   <td> DeepZ </td>
   <td> DeepPoly </td>
   <td> DeepZ </td> 
   <td> DeepPoly </td>
  </tr>

<tr>
   <td> MNIST</td>
   <td> ConvBig</td>
   <td> 0.1</td>
   <td> 97 </td>
   <td> 97 </td>
   <td> 5 </td> 
   <td> 50 </td>
</tr>


<tr>
   <td> </td>
   <td> ConvBig</td>
   <td> 0.2</td>
   <td> 79 </td>
   <td> 78 </td>
   <td> 7 </td> 
   <td> 61 </td>
</tr>

<tr>
   <td> </td>
   <td> ConvBig</td>
   <td> 0.3</td>
   <td> 37 </td>
   <td> 43 </td>
   <td> 17 </td> 
   <td> 88 </td>
</tr>

<tr>
   <td> </td>
   <td> ConvSuper</td>
   <td> 0.1</td>
   <td> 97 </td>
   <td> 97 </td>
   <td> 133 </td> 
   <td> 400 </td>
</tr>

<tr>
   <td> </td>
   <td> Skip</td>
   <td> 0.1</td>
   <td> 95 </td>
   <td> N/A </td>
   <td> 29 </td> 
   <td> N/A </td>
</tr>

<tr>
   <td> CIFAR10</td>
   <td> ConvBig</td>
   <td> 0.006</td>
   <td> 50 </td>
   <td> 52 </td>
   <td> 39 </td> 
   <td> 322 </td>
</tr>


<tr>
   <td> </td>
   <td> ConvBig</td>
   <td> 0.008</td>
   <td> 33 </td>
   <td> 40 </td>
   <td> 46 </td> 
   <td> 331 </td>
</tr>


</table>


The table below compares the timings of complete verification with ERAN for all ACASXu benchmarks. 


<table aligh="center">
  <tr>
    <td>Property</td>
    <td>Networks</td>
    <td colspan="1">% Average Runtime (s)</td>
  </tr>
  
  <tr>
   <td> 1</td>
   <td> all 45</td>
   <td> 15.5 </td>
  </tr>

<tr>
   <td> 2</td>
   <td> all 45</td>
   <td> 11.4 </td>
  </tr>

<tr>
   <td> 3</td>
   <td> all 45</td>
   <td> 1.9 </td>
  </tr>
  
<tr>
   <td> 4</td>
   <td> all 45</td>
   <td> 1.1 </td>
  </tr>

<tr>
   <td> 5</td>
   <td> 1_1</td>
   <td> 26 </td>
  </tr>

<tr>
   <td> 6</td>
   <td> 1_1</td>
   <td> 10 </td>
  </tr>
  
<tr>
   <td> 7</td>
   <td> 1_9</td>
   <td> 83 </td>
  </tr>

<tr>
   <td> 8</td>
   <td> 2_9</td>
   <td> 111 </td>
  </tr>

<tr>
   <td> 9</td>
   <td> 3_3</td>
   <td> 9 </td>
  </tr>
  
<tr>
   <td> 10</td>
   <td> 4_5</td>
   <td> 2.1 </td>
  </tr>

</table>


More experimental results can be found in [our technical report](https://arxiv.org/abs/2108.11651).

Contributors
--------------

* Yuyi Zhong (lead contact) - yuyizhong@comp.nus.edu.sg

* Tianzuo Luo - luotianzuo@u.nus.edu


<!-- License and Copyright
--------------------- -->

<!-- * Copyright (c) 2020 [Secure, Reliable, and Intelligent Systems Lab (SRI), Department of Computer Science ETH Zurich](https://www.sri.inf.ethz.ch/)
* Licensed under the [Apache License](https://www.apache.org/licenses/LICENSE-2.0) -->
