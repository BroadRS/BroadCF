# Broad Recommender System: An Efficient Nonlinear Collaborative Filtering Approach
This is our official implementation for the paper:

Ling Huang, Can-Rong Guan, Zhen-Wei Huang, Yuefang Gao, Yingjie Kuang,Chang-Dong Wang,and C. L. Philip Chen.Broad Recommender System: An Efficient Nonlinear Collaborative Filtering Approach.

In this paper, we have proposed a new neural network based recommender system called Broad Collaborative Filtering (BroadCF). Compared with the existing DNNs-based recommendation algorithms, the proposed BroadCF algorithm is also able to capture the nonlinear relationship between users and items, and hence generate very satisfactory recommendation performance. However, the superiority of BroadCF is that it is very efficient compared with the DNNs-based methods, i.e., it consumes relatively very short training time and only requires relatively small amount of data storage, in particular trainable parameter storage. The main advantage lies in designing a data preprocessing procedure to convert the original rating data into the form of (Rating collaborative vector, Rating one-hot vector), which is then feed into the very efficient BLS for rating prediction. 

Please cite our paper if you use our codes. Thanks!

# Environment Settings

We are using python 3.8.8 as the compilation environment.

# Example to run the codes
The instruction of commands has been clearly stated in the codes 

```
python init.py --dataset ml-1m
``` 
# Dataset
We provide six processed datasets: MovieLens 1 Million (ml-1m),A-Digital-Music, A-Grocery-and-Gourmet-Food, A-Patio-Lawn-and-Garden, A-Automotive, and A-Baby.
# Citation
