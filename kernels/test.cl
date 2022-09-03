__kernel void kernel1(int a){
    int idx = get_global_id(0);

    printf("线程：%d,a=%d\n",idx,a);
}