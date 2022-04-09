#!/bin/bash
gap=1

#Run our NUSPoly 
imgid=39
while (( $imgid <= 99))
do
    timeout -k 10 12600 /home/zhongyy/Verifier_nn/ERAN/eran_env/bin/python3.6 run_cascade.py --imageid $imgid --netname /home/zhongyy/Verifier_nn/RefineBBPoly/test_nns/mnist_nns/mnist_relu_9_200.onnx --dataset mnist --epsilon 0.015 --is_refinement True
    imgid=$((imgid+gap))
done

#Run our SMUPoly 
# imgid=2
# while (( $imgid <= 99))
# do
#     timeout -k 10 12600 /home/zhongyy/Verifier_nn/ERAN/eran_env/bin/python3.6 run_SMUPoly.py --imageid $imgid --netname /home/zhongyy/Verifier_nn/RefineBBPoly/test_nns/mnist_nns/ffnnRELU__Point_6_500.onnx --dataset mnist --epsilon 0.037 --is_refinement True
#     imgid=$((imgid+gap))
# done