__kernel void rbfKernel(__global float* set1,__global float* set2,__global float* result ,int n,int m,int size){


}

__kernel void testKernel(__global float* A,__global float* B,__global float* C){

    int idx = get_global_id(0);
    C[idx] = B[idx] + A[idx];
}