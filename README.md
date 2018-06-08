# Meta-Iterative Map-Reduce 
### Meta-Iterative Map-Reduce to perform Regression massively parallely on a cluster with MPI and CUDA for GPU and CPU-nodes support.
#### Authors: Shikhar Srivastava and Jawahar Reddy

### Details:
  - _CUDA-aware MPI_: Accelerate MPI by leveraging GPU compute through CUDA. https://devblogs.nvidia.com/introduction-cuda-aware-mpi/
  - _Iterative MapReduce_ : The Map-reduce paradigm was adapted for iterative operations, for example in Machine Learning model training. https://deeplearning4j.org/iterativereduce
  - _[Meta] Iterative MapReduce_: We (the authors) proposed a model that performs two "levels" of iterative map-reduce operations. It's gist is that each map-operation in the first level of map-reduce is a composite of another level of map-reduce operation. < Efficiency bounds are better this way >
  - _[Linear] Regression_: We train a Linear regression model as proof-of-concept to showcase the Meta Iterative Map-reduce paradigm.

<embed src="https://drive.google.com/viewerng/viewer?embedded=true&url=https://soilad.github.io/res/ppl_report.pdf" width="500" height="375" type='application/pdf'>

Download the Project Report <a href= 'https://soilad.github.io/res/ppl_report.pdf'> here </a>. 
