# Meta-Iterative Map-Reduce 

### _Implementation of the Meta-Iterative Map-Reduce algorithm to perform distributed & scalable training of a machine learning model on a GPU+CPU cluster using CUDA-aware-MPI._

#### Authors : Shikhar Srivastava & Jawahar Reddy

<embed src="https://drive.google.com/viewerng/viewer?embedded=true&url=https://soilad.github.io/res/ppl_report.pdf" width="500" height="375" type='application/pdf'>

Download the Project Report <a href= 'https://soilad.github.io/res/ppl_report.pdf'> here </a>. 



## What is Meta-Iterative Map Reduce?

Let's explain this using a bottom-up approach:

  - **_Map-Reduce_** is a programming model to parallelize computations of large tasks by parallelly solving sub-tasks. The sub-tasks are _mapped_ to multiple 'workers' that concurrently solve their parts and the outputs of the sub-tasks are _reduced_ back to form a solution to the primary task.

  - For tasks that benefit from iterations of computation, such as Machine Learning model training, the Map-Reduce operations are performed iteratively until a required solution is obtained. This programming model is therefore termed **_Iterative Map Reduce_**.

  - Now imagine, instead of diving the task into 1 level of sub-tasks, we continue to divide the sub-tasks further, into their own sub-_(sub)_-tasks that themselves follow a Map-reduce paradigm. 
This would mean that each 'worker' that was originally computing a sub-task, is now itself delegating work to a secondary level of workers. Done iteratively, we termed this composite of Map-reduce operations as _**Meta Iterative MapReduce**_. 

## Advantages of Meta-Iterative MapReduce

1. Effective speed-up of **`No. of Parallel MPI Processes ∗ No. of CUDA Kernel Threads`**

2. Using the Meta model, we can effectively leverage CUDA-aware-MPI. Thus,

    - _All operations that are required to carry out a message transfer i.e. a _send_ operation can be pipelined._
    
    - _Acceleration technologies like GPUDirect can be utilized by the MPI library transparently to the user._
    
3. Iterative MapReduce has significant applications for massively parallel, complex computations that are iteratively performed, such as modern Deep Learning applications, wherein both strict data store and floating-point operations requirements exist.

---

## Dependencies:

   - CUDA Toolkit (ver >= 7.0) 

   - Microsoft MPI or OpenMPI (tested on Microsoft MPI ver 8.0)

   - Nvidia Graphics card  [CUDA-supported GPU] 

## Installation:

  1. Clone the repository : `git clone https://github.com/soilad/Meta-Iterative-MapReduce.git `

  2. Ensure that the `cuda.h` header file is added to the compilation path in your IDE or mpicc compiler.

  3. Compile the kernel.cu file using the MPI compiler. 

      - For Microsoft MPI, 
  
            ```sh
            mpicc kernel.cu -o metamap
            ```
      - For Open MPI,
       
     Refer to https://www.open-mpi.org/faq/?category=runcuda & https://www.open-mpi.org/faq/?category=buildcuda
        
  4. Execute the compiled kernel code: `$ ./metamap`


## Key terms:

  - **_CUDA-aware MPI_**: Accelerate MPI by leveraging GPU compute through CUDA. https://devblogs.nvidia.com/introduction-cuda-aware-mpi/
  - **_Iterative MapReduce_** : The Map-reduce paradigm was adapted for iterative operations, for example in Machine Learning model training. https://deeplearning4j.org/iterativereduce
  - **_Meta Iterative MapReduce_** : We (the authors) proposed a model that performs two "levels" of iterative map-reduce operations. The gist is that each map-operation in the first level of map-reduce is a composite of another level of map-reduce operation. < Efficiency bounds are better this way >
  - **_[Linear] Regression_** : To showcase the improvement in model training speed, we perform distributed training of a Linear regression using the Meta-Iterative Map-Reduce programming model. 

---


