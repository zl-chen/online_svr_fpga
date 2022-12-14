#include <stdio.h>
#include <fstream>
#include <time.h>
#include <vector>
#include <math.h>
#include <iostream>
#include <string>
#include "onlinesvr.h"
#include "CL/opencl.h"
#include "AOCLUtils/aocl_utils.h"


#define NUM_ROWS 500
#define NUM_FEATURES 13

int * A,*C;

using namespace aocl_utils;
using namespace std;

const int msg_len = 1024;
char msg[msg_len];
cl_platform_id platform = NULL;
cl_device_id devices[2];
cl_context context;
cl_command_queue command_queue;
cl_program program;
cl_kernel kernel;
cl_mem A_buf;
cl_mem B_buf;
cl_mem C_buf;

void cleanup();

void load_data(double (*X)[NUM_FEATURES],double *Y){
    char temp;


    // 导入
    ifstream ifs("/home/shu_students/czl/online_svr_fpga/data/housing.csv",ifstream::in);




    // 用line保存从数据中中读取的每一行
    char line[1024];
    ifs.getline(line,1024);

    cout << line << endl;
    //cin >> temp;

    int idx = 0;
    
    // 循环读取每一行，将内容保存在X和Y数组中
    while(!ifs.eof()){

        char temp;
        for(int i=0;i<NUM_FEATURES;++i){
            ifs>>X[idx][i];
            printf("X[%d][%d]=%f\t",idx,i,X[idx][i]);
            ifs>>temp;
        }
        ifs>>Y[idx];
        printf("Y[%d]=%f\n",idx,Y[idx]);
        ++idx;
    }
    ifs.close();

    // 对X做数据清洗，将X数组归一化
    // 归一化：计算列特征的均值和标准差，然后用每个数值减去均值除以标准差。归一化的目的是将特征数据变成均值为0，标准差为1的分布。
    vector<double> means;
    for(int i=0;i<NUM_FEATURES;++i)
    {
        // 计算
        double mean = 0;
        for(int j=0;j<NUM_ROWS;++j){
            mean += X[j][i];
        }
        mean = mean/NUM_ROWS;

        means.push_back(mean);
    }

    // 计算标准差
    vector<double> sd_vec;
    for(int i=0;i<NUM_FEATURES;++i){
        double sd = 0;
        
        for(int j=0;j<NUM_ROWS;++j){
            sd += (X[j][i] - means[i]) * (X[j][i] - means[i]);
        }

        sd = sqrt(sd/NUM_ROWS);

        sd_vec.push_back(sd);
    }


    // 进行归一化
    for(int i=0;i<NUM_FEATURES;++i){
        double mean = means[i];
        double sd = sd_vec[i];

        for(int j=0;j<NUM_ROWS;++j){
            X[j][i] = (X[j][i] - mean)/sd;
        }
    }

}


void train(){

    char temp;
   // cout << "Enter train" << endl;cin >> temp;
    setCwdToExeDir();
    
    cl_int status;

    // 建立平台
    status = clGetPlatformIDs(1,&platform,NULL);
    checkError(status,"Faile to find platform");

    // 建立设备
    status = clGetDeviceIDs(platform,CL_DEVICE_TYPE_ALL,2,devices,NULL);
    checkError(status,"FAILED to find devices");

    // 使用设备1
    cl_device_id device = devices[0];

    // 建立context
    context = clCreateContext(NULL,1,&device,NULL,NULL,&status);
    checkError(status,"Failed to create context");

    // 建立队列
    command_queue = clCreateCommandQueue(context,device,CL_QUEUE_PROFILING_ENABLE,&status);
    checkError(status,"Failed to create commmand queue");
 

    // 建立program
    program = createProgramFromBinary(context,"rbfKernel.aocx",&device,1);
    status = clBuildProgram(program,0,NULL,NULL,NULL,NULL);
    checkError(status,"Failed to create program");


    // 创建kernel对象
    kernel = clCreateKernel(program,"rbfKernel",&status);
    checkError(status,"Failed to create kernel");

    // 创建buffer
    cl_mem A_buf = clCreateBuffer(context,CL_MEM_READ_ONLY,5*sizeof(float),NULL,&status);
    cl_mem B_buf = clCreateBuffer(context,CL_MEM_READ_ONLY,5*sizeof(float),NULL,&status);
    cl_mem C_buf = clCreateBuffer(context,CL_MEM_WRITE_ONLY,5*sizeof(float),NULL,&status);
    
    


    
    double X[NUM_ROWS][NUM_FEATURES];
    double Y[NUM_ROWS];
    load_data(X,Y);

    printf("dsafasdfasdfasdfasdfs\n");
    //cin >> temp;


    for(int i=0;i<NUM_FEATURES;++i){
        printf("X[%d]=%lf\t",i,X[0][i]);
    }
    printf("Y=%f\n",Y[0]);
   

    //cin >> temp;

    OnlineSVR online_svr(13,100,0.1,0.1,0.5,command_queue,kernel,context);

    int train_num = 400;


    clock_t start = clock();
    for(int i=0;i<train_num;++i){
        
   
        vector<double> xVec(begin(X[i]),end(X[i]));
        online_svr.learn(xVec,Y[i]);
        
         vector<vector<double>> newX;

        vector<double> xVec2(begin(X[i+1]),end(X[i+1]));

        newX.push_back(xVec);


        cout << i << "   " << online_svr.predict(newX)[0]  << "\t" << Y[i+1]<< endl ;

    }
    clock_t endTime = clock();

    cout << double(endTime-start)/CLOCKS_PER_SEC << "s" << endl;

    
    // 计算后续100个值的mse
    int mse_num = 100;
    double mse = 0.0;
    for(int i=train_num;i<train_num+mse_num;++i){
        vector<vector<double>> newX;
        vector<double> xVec(begin(X[i]),end(X[i]));
        newX.push_back(xVec);

        double y_pre = online_svr.predict(newX)[0];

        mse += (y_pre-Y[i])*(y_pre-Y[i]);

    }

    // 
    cout << mse/mse_num << endl;
    

    if(kernel){
        clReleaseKernel(kernel);
    }

    if(program){
        clReleaseProgram(program);
    }

    if(command_queue){
        clReleaseCommandQueue(command_queue);
    }
    
    if(context){
        clReleaseContext(context);
    }

}




void openCLInit(){


    /*
    for(int i=0;i<N;++i){
        for(int j=0;j<N;++j){
            A[i * N + j] = 1;B[i * N + j] = 2;
            C[i * N + j] = 0;
        }
    }
    */


    cl_int status;

    setCwdToExeDir();

    // 查平台
    status = clGetPlatformIDs(1,&platform,NULL);
    checkError(status,"Failed to find a platform");


    // 查设备
    status = clGetDeviceIDs(platform,CL_DEVICE_TYPE_ALL,2,devices,NULL);
    checkError(status,"Failed to find a device");

    status = clGetDeviceInfo(devices[0],CL_DEVICE_NAME,msg_len,msg,NULL);
    checkError(status,"Failed to fisnd device info");

   // 创建context
   context = clCreateContext(NULL,1,devices,NULL,NULL,&status);
   checkError(status,"Failed to create context");

   // 创建command queue 
   command_queue = clCreateCommandQueue(context,devices[0],CL_QUEUE_PROFILING_ENABLE,&status);
   checkError(status,"Faile to create command queue");

    // 创建并build program
    program = createProgramFromBinary(context,"matrix.aocx",devices,1);
    status = clBuildProgram(program,0,NULL,NULL,NULL,NULL);
    checkError(status,"Failed to build program");

    // 创建kernel
    kernel = clCreateKernel(program,"matMul",&status);
    checkError(status,"Failed to create kernel");

    // 创建buffer
    A_buf = clCreateBuffer(context,CL_MEM_READ_ONLY,1,NULL,&status);
    C_buf = clCreateBuffer(context,CL_MEM_READ_ONLY,1,NULL,&status);


    // 设置kernel参数
    status = clSetKernelArg(kernel,0,sizeof(cl_mem),&A_buf);

    checkError(status,"Failed to set arg");


    clock_t start = clock();
    // 传送数据
    status = clEnqueueWriteBuffer(command_queue,A_buf,CL_TRUE,0,sizeof(int)*1,A,0,NULL,NULL);

    // 启动kernel
    size_t gSize[3] = {5,1,1};
    size_t lSize[3] = {5,1,1};
    status = clEnqueueNDRangeKernel(command_queue,kernel,1,NULL,gSize,lSize,0,NULL,NULL);
    checkError(status,"Failed to launch kernel");

    // 传回数据
    status = clEnqueueReadBuffer(command_queue,C_buf,CL_TRUE,0,sizeof(int)*1,C,0,NULL,NULL);
    
    // 等待队列中的kernel完成
    status = clFinish(command_queue);
    checkError(status,"Failed to finish");



    


    // 回收内存
    if(kernel) {
        clReleaseKernel(kernel);  
    }
    if(program) {
        clReleaseProgram(program);
    }
    if(command_queue) {
        clReleaseCommandQueue(command_queue);
    }
    if(context) {
        clReleaseContext(context);
    }
       
    delete A;
    delete C;

}

int main(int argc ,char** argv)
{
    
    train();
    return 0;
}


void cleanup(){
    return ;
}