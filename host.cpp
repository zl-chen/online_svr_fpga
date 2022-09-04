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


#define NUM_ROWS 1222
#define NUM_FEATURES 3

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

    // 导入
    ifstream ifs("/home/shu_students/czl/online_svr_fpga/data.csv",ifstream::in);

    char line[1024];
    ifs.getline(line,1024);


    int idx = 0;
    
    while(!ifs.eof()){

        char temp;
        for(int i=0;i<3;++i){
            ifs>>X[idx][i];
            ifs>>temp;
        }
        ifs>>Y[idx];
        ++idx;
    }
    ifs.close();

    // 数据清洗
    vector<double> means;
    for(int i=0;i<NUM_FEATURES;++i){
        double mean = 0;
        for(int j=0;j<NUM_ROWS;++j){
            mean += X[j][i];
        }
        mean = mean/NUM_ROWS;

        means.push_back(mean);
    }
    vector<double> sd_vec;
    for(int i=0;i<NUM_FEATURES;++i){
        double sd = 0;
        
        for(int j=0;j<NUM_ROWS;++j){
            sd += (X[j][i] - means[i]) * (X[j][i] - means[i]);
        }

        sd = sqrt(sd/NUM_ROWS);

        sd_vec.push_back(sd);
    }

    for(int i=0;i<NUM_FEATURES;++i){
        double mean = means[i];
        double sd = sd_vec[i];

        for(int j=0;j<NUM_ROWS;++j){
            X[j][i] = (X[j][i] - mean)/sd;
        }
    }

}


void train(){
    setCwdToExeDir();
    
    cout << "test" << endl;

    cl_int status;

    status = clGetPlatformIDs(1,&platform,NULL);
    checkError(status,"Faile to find platform");

    char line[1024];
    clGetPlatformInfo(platform,CL_PLATFORM_NAME,1024,line,NULL);

    cout << "Name: " << line << endl;


    cl_uint num;
    status = clGetDeviceIDs(platform,CL_DEVICE_TYPE_ALL,2,devices,&num);
    cout << num << " devices" << endl;
    checkError(status,"FAILED to find devices");



    context = clCreateContext(NULL,1,devices,NULL,NULL,&status);
    checkError(status,"Failed to create context");



    // 建队列
    command_queue = clCreateCommandQueue(context,devices[0],CL_QUEUE_PROFILING_ENABLE,&status);
    checkError(status,"Failed to create commmand queue");
 

    // 建立program
    program = createProgramFromBinary(context,"rbfKernel.aocx",devices,1);
    status = clBuildProgram(program,0,NULL,NULL,NULL,NULL);
    checkError(status,"Failed to create program");


    // 创建kernel对象
    kernel = clCreateKernel(program,"testKernel",&status);
    checkError(status,"Failed to create kernel");
    


    
    double X[NUM_ROWS][NUM_FEATURES];
    double Y[NUM_ROWS];
    load_data(X,Y);
     cout << "Enter train func####2################" << endl;

    OnlineSVR online_svr(3,143,0.1,0.1,0.5,command_queue,kernel);

    int train_num = 300;

    
    for(int i=0;i<train_num;++i){
   
        vector<double> xVec(begin(X[i]),end(X[i]));
        //online_svr.learn(xVec,Y[i]);

        online_svr.testOpenCL();

        
         vector<vector<double>> newX;

        vector<double> xVec2(begin(X[i+1]),end(X[i+1]));

        newX.push_back(xVec);


        //cout << i << "   " << online_svr.predict(newX)[0] << endl ;

    }

    double mse = 0.0;
    for(int i=train_num;i<train_num+100;++i){
        vector<vector<double>> newX;
        vector<double> xVec(begin(X[i]),end(X[i]));
        newX.push_back(xVec);

        double y_pre = online_svr.predict(newX)[0];

        mse += (y_pre-Y[i])*(y_pre-Y[i]);

    }


    cout << mse/100 << endl;
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