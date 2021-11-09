BBPoly instantiated on ERAN
========

BBPoly is a scalabel and module robustness verification tool which is instantiated on DeepPoly domain in ETH Robustness Analyzer for Neural Networks (ERAN). BBPoly provides incomplete verification of MNIST, CIFAR10 based networks can be tuned to achieve trade-off between precision and scalability (see recommended configuration settings at the bottom). 

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

L_oo Specification
```
python3 . --netname ../nets/pytorch/mnist/convBig__DiffAI.pyt --epsilon 0.1 --domain deepzono --dataset mnist
```

will evaluate the local robustness of the MNIST convolutional network (upto 35K neurons) with ReLU activation trained using DiffAI on the 100 MNIST test images. In the above setting, epsilon=0.1 and the domain used by our analyzer is the deepzono domain. Our analyzer will print the following:

* 'Verified safe' for an image when it can prove the robustness of the network 

* 'Verified unsafe' for an image for which it can provide a concrete adversarial example

* 'Failed' when it cannot. 

* It will also print an error message when the network misclassifies an image.

* the timing in seconds.

* The ratio of images on which the network is robust versus the number of images on which it classifies correctly.
 

Zonotope Specification
```
python3 . --netname ../nets/pytorch/mnist/convBig__DiffAI.pyt --zonotope some_path/zonotope_example.txt --domain deepzono 
```

will check if the Zonotope specification specified in "zonotope_example" holds for the network and will output "Verified safe", "Verified unsafe" or "Failed" along with the timing.

Similarly, for the ACAS Xu networks, ERAN will output whether the property has been verified along with the timing.


ACASXu Specification
```
python3 . --netname ../data/acasxu/nets/ACASXU_run2a_3_3_batch_2000.onnx --dataset acasxu --domain deepzono  --specnumber 9
```
will run DeepZ for analyzing property 9 of ACASXu benchmarks. The ACASXU networks are in data/acasxu/nets directory and the one chosen for a given property is defined in the Reluplex paper. 

Geometric analysis

```
python3 . --netname ../nets/pytorch/mnist/convBig__DiffAI.pyt --geometric --geometric_config ../deepg/examples/example1/config.txt --num_params 1 --dataset mnist
```
will on the fly generate geometric perturbed images and evaluate the network against them. For more information on the geometric configuration file please see [Format of the configuration file in DeepG](https://github.com/eth-sri/deepg#format-of-configuration-file).


```
python3 . --netname ../nets/pytorch/mnist/convBig__DiffAI.pyt --geometric --data_dir ../deepg/examples/example1/ --num_params 1 --dataset mnist --attack
```
will evaluate the generated geometric perturbed images in the given data_dir and also evaluate generated attack images.


Recommended Configuration for Scalable Complete Verification
---------------------------------------------------------------------------------------------
Use the "deeppoly" or "deepzono" domain with "--complete True" option


Recommended Configuration for More Precise but relatively expensive Incomplete Verification
----------------------------------------------------------------------------------------------
Use the "refinepoly" domain with "--use_milp True", "--sparse_n 12", "--refine_neurons", "timeout_milp 10", and "timeout_lp 10" options

Recommended Configuration for Faster but relatively imprecise Incomplete Verification
-----------------------------------------------------------------------------------------------
Use the "deeppoly" domain


Publications
-------------
*  [Certifying Geometric Robustness of Neural Networks](https://www.sri.inf.ethz.ch/publications/balunovic2019geometric)

   Mislav Balunovic,  Maximilian Baader, Gagandeep Singh, Timon Gehr,  Martin Vechev
   
   NeurIPS 2019.


*  [Beyond the Single Neuron Convex Barrier for Neural Network Certification](https://www.sri.inf.ethz.ch/publications/singh2019krelu).
    
    Gagandeep Singh, Rupanshu Ganvir, Markus Püschel, and Martin Vechev. 
   
    NeurIPS 2019.

*  [Boosting Robustness Certification of Neural Networks](https://www.sri.inf.ethz.ch/publications/singh2019refinement).

    Gagandeep Singh, Timon Gehr, Markus Püschel, and Martin Vechev. 

    ICLR 2019.


*  [An Abstract Domain for Certifying Neural Networks](https://www.sri.inf.ethz.ch/publications/singh2019domain).

    Gagandeep Singh, Timon Gehr, Markus Püschel, and Martin Vechev. 

    POPL 2019.


*  [Fast and Effective Robustness Certification](https://www.sri.inf.ethz.ch/publications/singh2018effective). 

    Gagandeep Singh, Timon Gehr, Matthew Mirman, Markus Püschel, and Martin Vechev. 

    NeurIPS 2018.




Neural Networks and Datasets
---------------

We provide a number of pretrained MNIST and CIAFR10 defended and undefended feedforward and convolutional neural networks with ReLU, Sigmoid and Tanh activations trained with the PyTorch and TensorFlow frameworks. The adversarial training to obtain the defended networks is performed using PGD and [DiffAI](https://github.com/eth-sri/diffai). 

| Dataset  |   Model  |  Type   | #units | #layers| Activation | Training Defense| Download |
| :-------- | :-------- | :-------- | :-------------| :-------------| :------------ | :------------- | :---------------:|
| MNIST   | 3x50 | fully connected | 110 | 3    | ReLU | None | [:arrow_down:](https://files.sri.inf.ethz.ch/eran/nets/tensorflow/mnist/mnist_relu_3_50.tf)|
|         | 3x100 | fully connected | 210 | 3    | ReLU | None | [:arrow_down:](https://files.sri.inf.ethz.ch/eran/nets/tensorflow/mnist/mnist_relu_3_100.tf)|
|         | 5x100 | fully connected | 510 | 5    | ReLU | DiffAI | [:arrow_down:](https://files.sri.inf.ethz.ch/eran/nets/tensorflow/mnist/mnist_relu_5_100.tf)|
|         | 6x100 | fully connected | 510 | 6    | ReLU | None | [:arrow_down:](https://files.sri.inf.ethz.ch/eran/nets/tensorflow/mnist/mnist_relu_6_100.tf)|
|         | 9x100 | fully connected | 810 | 9    | ReLU | None | [:arrow_down:](https://files.sri.inf.ethz.ch/eran/nets/tensorflow/mnist/mnist_relu_9_100.tf)|
|         | 6x200 | fully connected | 1,010 | 6   | ReLU | None | [:arrow_down:](https://files.sri.inf.ethz.ch/eran/nets/tensorflow/mnist/mnist_relu_6_200.tf)|
|         | 9x200 | fully connected | 1,610 | 9   | ReLU | None | [:arrow_down:](https://files.sri.inf.ethz.ch/eran/nets/tensorflow/mnist/mnist_relu_9_200.tf)|
|         | 6x500 | fully connected | 3,000 | 6   | ReLU | None | [:arrow_down:](https://files.sri.inf.ethz.ch/eran/nets/pytorch/mnist/ffnnRELU__Point_6_500.pyt)|
|         | 6x500 | fully connected | 3,000 | 6   | ReLU  | PGD &epsilon;=0.1 | [:arrow_down:](https://files.sri.inf.ethz.ch/eran/nets/pytorch/mnist/ffnnRELU__PGDK_w_0.1_6_500.pyt)|
|         | 6x500 | fully connected | 3,000 |  6  | ReLU | PGD &epsilon;=0.3 | [:arrow_down:](https://files.sri.inf.ethz.ch/eran/nets/pytorch/mnist/ffnnRELU__PGDK_w_0.3_6_500.pyt)|
|         | 6x500 | fully connected | 3,000  | 6   | Sigmoid | None | [:arrow_down:](https://files.sri.inf.ethz.ch/eran/nets/pytorch/mnist/ffnnSIGMOID__Point_6_500.pyt)|
|         | 6x500 | fully connected | 3,000 |  6  | Sigmoid | PGD &epsilon;=0.1 | [:arrow_down:](https://files.sri.inf.ethz.ch/eran/nets/pytorch/mnist/ffnnSIGMOID__PGDK_w_0.1_6_500.pyt)|
|         | 6x500 | fully connected | 3,000 | 6   | Sigmoid | PGD &epsilon;=0.3 | [:arrow_down:](https://files.sri.inf.ethz.ch/eran/nets/pytorch/mnist/ffnnSIGMOID__PGDK_w_0.3_6_500.pyt)|
|         | 6x500 | fully connected | 3,000 | 6 |    Tanh | None | [:arrow_down:](https://files.sri.inf.ethz.ch/eran/nets/pytorch/mnist/ffnnTANH__Point_6_500.pyt)|
|         | 6x500 |  fully connected| 3,000 | 6   | Tanh | PGD &epsilon;=0.1 | [:arrow_down:](https://files.sri.inf.ethz.ch/eran/nets/pytorch/mnist/ffnnTANH__PGDK_w_0.1_6_500.pyt)|
|         | 6x500 | fully connected | 3,000 | 6   |  Tanh | PGD &epsilon;=0.3 | [:arrow_down:](https://files.sri.inf.ethz.ch/eran/nets/pytorch/mnist/ffnnTANH__PGDK_w_0.3_6_500.pyt)|
|         | 4x1024 | fully connected | 3,072 | 4   | ReLU | None | [:arrow_down:](https://files.sri.inf.ethz.ch/eran/nets/tensorflow/mnist/mnist_relu_4_1024.tf)|
|         |  ConvSmall | convolutional | 3,604 | 3  | ReLU | None | [:arrow_down:](https://files.sri.inf.ethz.ch/eran/nets/pytorch/mnist/convSmallRELU__Point.pyt)|
|         |  ConvSmall | convolutional | 3,604 | 3  | ReLU | PGD | [:arrow_down:](https://files.sri.inf.ethz.ch/eran/nets/pytorch/mnist/convSmallRELU__PGDK.pyt) |
|         |  ConvSmall | convolutional | 3,604 | 3  | ReLU | DiffAI | [:arrow_down:](https://files.sri.inf.ethz.ch/eran/nets/pytorch/mnist/convSmallRELU__DiffAI.pyt) |
|         | ConvMed | convolutional | 5,704 | 3  | ReLU | None | [:arrow_down:](https://files.sri.inf.ethz.ch/eran/nets/pytorch/mnist/convMedGRELU__Point.pyt) |
|         | ConvMed | convolutional | 5,704 | 3   | ReLU | PGD &epsilon;=0.1 | [:arrow_down:](https://files.sri.inf.ethz.ch/eran/nets/pytorch/mnist/convMedGRELU__PGDK_w_0.1.pyt) |
|         | ConvMed | convolutional | 5,704 | 3   | ReLU | PGD &epsilon;=0.3 | [:arrow_down:](https://files.sri.inf.ethz.ch/eran/nets/pytorch/mnist/convMedGRELU__PGDK_w_0.3.pyt) |
|         | ConvMed | convolutional | 5,704 | 3   | Sigmoid | None | [:arrow_down:](https://files.sri.inf.ethz.ch/eran/nets/pytorch/mnist/convMedGSIGMOID__Point.pyt) |
|         | ConvMed | convolutional | 5,704 | 3   | Sigmoid | PGD &epsilon;=0.1 | [:arrow_down:](https://files.sri.inf.ethz.ch/eran/nets/pytorch/mnist/convMedGSIGMOID__PGDK_w_0.1.pyt) | 
|         | ConvMed | convolutional | 5,704 | 3   | Sigmoid | PGD &epsilon;=0.3 | [:arrow_down:](https://files.sri.inf.ethz.ch/eran/nets/pytorch/mnist/convMedGSIGMOID__PGDK_w_0.3.pyt) | 
|         | ConvMed | convolutional | 5,704 | 3   | Tanh | None | [:arrow_down:](https://files.sri.inf.ethz.ch/eran/nets/pytorch/mnist/convMedGTANH__Point.pyt) |
|         | ConvMed | convolutional | 5,704 | 3   | Tanh | PGD &epsilon;=0.1 | [:arrow_down:](https://files.sri.inf.ethz.ch/eran/nets/pytorch/mnist/convMedGTANH__PGDK_w_0.1.pyt) | 
|         | ConvMed | convolutional | 5,704 | 3   |  Tanh | PGD &epsilon;=0.3 | [:arrow_down:](https://files.sri.inf.ethz.ch/eran/nets/pytorch/mnist/convMedGTANH__PGDK_w_0.3.pyt) |
|         | ConvMaxpool | convolutional | 13,798 | 9 | ReLU | None | [:arrow_down:](https://files.sri.inf.ethz.ch/eran/nets/tensorflow/mnist/mnist_conv_maxpool.tf)|
|         | ConvBig | convolutional | 48,064 | 6  | ReLU | DiffAI | [:arrow_down:](https://files.sri.inf.ethz.ch/eran/nets/pytorch/mnist/convBigRELU__DiffAI.pyt) |
|         | ConvSuper | convolutional | 88,544 | 6  | ReLU | DiffAI | [:arrow_down:](https://files.sri.inf.ethz.ch/eran/nets/pytorch/mnist/convSuperRELU__DiffAI.pyt) |
|         | Skip      | Residual | 71,650 | 9 | ReLU | DiffAI | [:arrow_down:](https://files.sri.inf.ethz.ch/eran/nets/pytorch/mnist/skip__DiffAI.pyt) |
| CIFAR10 | 4x100 | fully connected | 410 | 4 | ReLU | None | [:arrow_down:](https://files.sri.inf.ethz.ch/eran/nets/tensorflow/cifar/cifar_relu_4_100.tf) |
|         | 6x100 | fully connected | 610 | 6 | ReLU | None | [:arrow_down:](https://files.sri.inf.ethz.ch/eran/nets/tensorflow/cifar/cifar_relu_6_100.tf) |
|         | 9x200 | fully connected | 1,810 | 9 | ReLU | None | [:arrow_down:](https://files.sri.inf.ethz.ch/eran/nets/tensorflow/cifar/cifar_relu_9_200.tf) |
|         | 6x500 | fully connected | 3,000 | 6   | ReLU | None | [:arrow_down:](https://files.sri.inf.ethz.ch/eran/nets/pytorch/cifar/ffnnRELU__Point_6_500.pyt)|
|         | 6x500 | fully connected | 3,000 | 6   | ReLU | PGD &epsilon;=0.0078 | [:arrow_down:](https://files.sri.inf.ethz.ch/eran/nets/pytorch/cifar/ffnnRELU__PGDK_w_0.0078_6_500.pyt)|
|         | 6x500 | fully connected | 3,000 | 6   | ReLU | PGD &epsilon;=0.0313 | [:arrow_down:](https://files.sri.inf.ethz.ch/eran/nets/pytorch/cifar/ffnnRELU__PGDK_w_0.0313_6_500.pyt)| 
|         | 6x500 | fully connected | 3,000 | 6   | Sigmoid | None | [:arrow_down:](https://files.sri.inf.ethz.ch/eran/nets/pytorch/cifar/ffnnSIGMOID__Point_6_500.pyt)|
|         | 6x500 | fully connected | 3,000 | 6   | Sigmoid | PGD &epsilon;=0.0078 | [:arrow_down:](https://files.sri.inf.ethz.ch/eran/nets/pytorch/cifar/ffnnSIGMOID__PGDK_w_0.0078_6_500.pyt)|
|         | 6x500 | fully connected | 3,000 | 6   | Sigmoid | PGD &epsilon;=0.0313 | [:arrow_down:](https://files.sri.inf.ethz.ch/eran/nets/pytorch/cifar/ffnnSIGMOID__PGDK_w_0.0313_6_500.pyt)| 
|         | 6x500 | fully connected | 3,000 | 6   | Tanh | None | [:arrow_down:](https://files.sri.inf.ethz.ch/eran/nets/pytorch/cifar/ffnnTANH__Point_6_500.pyt)|
|         | 6x500 | fully connected | 3,000 | 6   | Tanh | PGD &epsilon;=0.0078 | [:arrow_down:](https://files.sri.inf.ethz.ch/eran/nets/pytorch/cifar/ffnnTANH__PGDK_w_0.0078_6_500.pyt)|
|         | 6x500 | fully connected | 3,000 | 6   | Tanh | PGD &epsilon;=0.0313 |  [:arrow_down:](https://files.sri.inf.ethz.ch/eran/nets/pytorch/cifar/ffnnTANH__PGDK_w_0.0313_6_500.pyt)| 
|         | 7x1024 | fully connected | 6,144 | 7 | ReLU | None | [:arrow_down:](https://files.sri.inf.ethz.ch/eran/nets/tensorflow/cifar/cifar_relu_7_1024.tf) |
|         | ConvSmall | convolutional | 4,852 | 3 | ReLU | None | [:arrow_down:](https://files.sri.inf.ethz.ch/eran/nets/pytorch/cifar/convSmallRELU__Point.pyt)|
|         | ConvSmall   | convolutional  | 4,852 | 3  | ReLU  | PGD | [:arrow_down:](https://files.sri.inf.ethz.ch/eran/nets/pytorch/cifar/convSmallRELU__PGDK.pyt)|
|         | ConvSmall  | convolutional | 4,852 | 3  | ReLU | DiffAI | [:arrow_down:](https://files.sri.inf.ethz.ch/eran/nets/pytorch/cifar/convSmallRELU__DiffAI.pyt)|
|         | ConvMed | convolutional | 7,144 | 3 | ReLU | None | [:arrow_down:](https://files.sri.inf.ethz.ch/eran/nets/pytorch/cifar/convMedGRELU__Point.pyt) |
|         | ConvMed | convolutional | 7,144 | 3   | ReLU | PGD &epsilon;=0.0078 | [:arrow_down:](https://files.sri.inf.ethz.ch/eran/nets/pytorch/cifar/convMedGRELU__PGDK_w_0.0078.pyt) |
|         | ConvMed | convolutional | 7,144 | 3   | ReLU | PGD &epsilon;=0.0313 | [:arrow_down:](https://files.sri.inf.ethz.ch/eran/nets/pytorch/cifar/convMedGRELU__PGDK_w_0.0313.pyt) | 
|         | ConvMed | convolutional | 7,144 | 3   | Sigmoid | None | [:arrow_down:](https://files.sri.inf.ethz.ch/eran/nets/pytorch/cifar/convMedGSIGMOID__Point.pyt) |
|         | ConvMed | convolutional | 7,144 | 3   | Sigmoid | PGD &epsilon;=0.0078 | [:arrow_down:](https://files.sri.inf.ethz.ch/eran/nets/pytorch/cifar/convMedGSIGMOID__PGDK_w_0.0078.pyt) |
|         | ConvMed | convolutional | 7,144 | 3   | Sigmoid | PGD &epsilon;=0.0313 | [:arrow_down:](https://files.sri.inf.ethz.ch/eran/nets/pytorch/cifar/convMedGSIGMOID__PGDK_w_0.0313.pyt) | 
|         | ConvMed | convolutional | 7,144 | 3   | Tanh | None | [:arrow_down:](https://files.sri.inf.ethz.ch/eran/nets/pytorch/cifar/convMedGTANH__Point.pyt) |
|         | ConvMed | convolutional | 7,144 | 3   | Tanh | PGD &epsilon;=0.0078 | [:arrow_down:](https://files.sri.inf.ethz.ch/eran/nets/pytorch/cifar/convMedGTANH__PGDK_w_0.0078.pyt) |
|         | ConvMed | convolutional | 7,144 | 3   | Tanh | PGD &epsilon;=0.0313 | [:arrow_down:](https://files.sri.inf.ethz.ch/eran/nets/pytorch/cifar/convMedGTANH__PGDK_w_0.0313.pyt) |  
|         | ConvMaxpool | convolutional | 53,938 | 9 | ReLU | None | [:arrow_down:](https://files.sri.inf.ethz.ch/eran/nets/tensorflow/cifar/cifar_conv_maxpool.tf)|
|         | ConvBig | convolutional | 62,464 | 6 | ReLU | DiffAI | [:arrow_down:](https://files.sri.inf.ethz.ch/eran/nets/pytorch/cifar/convBigRELU__DiffAI.pyt) | 
|         | ResNet18 | Residual | 558K | 18 | ReLU | DiffAI | [:arrow_down:](https://files.sri.inf.ethz.ch/eran/nets/pytorch/cifar/ResNet18_DiffAI.pyt) | 

We provide the first 100 images from the testset of both MNIST and CIFAR10 datasets in the 'data' folder. Our analyzer first verifies whether the neural network classifies an image correctly before performing robustness analysis. In the same folder, we also provide ACAS Xu networks and property specifications.

Experimental Results
--------------
We ran our experiments for the feedforward networks on a 3.3 GHz 10 core Intel i9-7900X Skylake CPU with a main memory of 64 GB whereas our experiments for the convolutional networks were run on a 2.6 GHz 14 core Intel Xeon CPU E5-2690 with 512 GB of main memory. We first compare the precision and performance of DeepZ and DeepPoly vs [Fast-Lin](https://github.com/huanzhang12/CertifiedReLURobustness) on the MNIST 6x100 network in single threaded mode. It can be seen that DeepZ has the same precision as Fast-Lin whereas DeepPoly is more precise while also being faster.

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


More experimental results can be found in our papers.

Contributors
--------------

* [Gagandeep Singh](https://www.sri.inf.ethz.ch/people/gagandeep) (lead contact) - gsingh@inf.ethz.ch

* Jonathan Maurer - maurerjo@student.ethz.ch

* Christoph Müller (contact for GPU version of DeepPoly) - christoph.mueller@inf.ethz.ch

* [Matthew Mirman](https://www.mirman.com) - matt@mirman.com

* [Timon Gehr](https://www.sri.inf.ethz.ch/tg.php) - timon.gehr@inf.ethz.ch
 
* Adrian Hoffmann - adriahof@student.ethz.ch

* Mislav Balunovic (https://www.sri.inf.ethz.ch/people/mislav) - mislav.balunovic@inf.ethz.ch

* Maximilian Baader (https://www.sri.inf.ethz.ch/people/max) - mbaader@inf.ethz.ch

* [Petar Tsankov](https://www.sri.inf.ethz.ch/people/petar) - petar.tsankov@inf.ethz.ch

* [Dana Drachsler Cohen](https://www.sri.inf.ethz.ch/people/dana) - dana.drachsler@inf.ethz.ch 

* [Markus Püschel](https://acl.inf.ethz.ch/people/markusp/) - pueschel@inf.ethz.ch

* [Martin Vechev](https://www.sri.inf.ethz.ch/vechev.php) - martin.vechev@inf.ethz.ch

License and Copyright
---------------------

* Copyright (c) 2020 [Secure, Reliable, and Intelligent Systems Lab (SRI), Department of Computer Science ETH Zurich](https://www.sri.inf.ethz.ch/)
* Licensed under the [Apache License](https://www.apache.org/licenses/LICENSE-2.0)
