__kernel void matMul(__global int * A,__global int * B,__global int * C,__local int * row,int N){
    int global_idx = get_global_id(0);

    int local_idx = get_local_id(0);


    int x = global_idx/N;
    int y = global_idx%N;

    if(y == 0){
        for(int i=0;i<N;++i){
            row[i] = A[x * N + i];
        }
    }


    int result  = 0;

    for(int i=0;i<N;++i){
        result += row[i] * B[i * N + y ];
    }

    C[x * N + y] = result;

}