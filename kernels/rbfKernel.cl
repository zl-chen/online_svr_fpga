
__kernel void rbfKernel(__global float* set1,__global float* set2,__global float* result ,int n,int m,int size,float kernelParam){
    // set1 n行 
    // set2 m行
    // 每行 size个元素

    int idx = get_global_id(0);
    int row = idx/m;
    int col = idx%m;

    float ans = 0.0;


    // 执行rbf计算

    // 平方和
    float squareSum = 0.0;
    for(int i=0;i<size;++i){
       squareSum += (set1[row * size + i] - set2[col * size + i]) * (set1[row * size + i] - set2[col * size + i]);
        
    }
    // 平方和 kernelParam乘法
    squareSum *= -kernelParam;
 
    result[idx] = squareSum;

}

__kernel void testKernel(__global float* A,__global float* B,__global float* C){

    int idx = get_global_id(0);
    C[idx] = B[idx] + A[idx];
}