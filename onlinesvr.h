#ifndef _ONLINE_SVR_H
#define _ONLINE_SVR_H

#include <vector>
#include <stdio.h>
#include <math.h>
#include <algorithm>
#include <string>
#include <fstream>
#include <ios>
#include "CL/opencl.h"
#include "AOCLUtils/aocl_utils.h"

using std::string;
using std::vector;
using std::ofstream;
using std::ios;
using namespace aocl_utils;

class OnlineSVR{
public:
    OnlineSVR(int numFeatures,int C,double eps,double kernelParam,double bias,cl_command_queue command_queue,cl_kernel kernel,
    cl_mem A_buf,cl_mem B_buf,cl_mem C_buf)
    :numFeatures(numFeatures),C(C),eps(eps),kernelParam(kernelParam),bias(bias),command_queue(command_queue),
    kernel(kernel),A_buf(A_buf),B_buf(B_buf),C_buf(C_buf){
        this->numSamplesTrained = 0;
    }
    void testOpenCL(){

        // 状态指示
        cl_uint status;


        // 设置kernel参数
        status = clSetKernelArg(kernel,0,sizeof(cl_mem),&A_buf);
        status = clSetKernelArg(kernel,1,sizeof(cl_mem),&B_buf);
        status = clSetKernelArg(kernel,2,sizeof(cl_mem),&C_buf);

        // 主机发送数据
        float A[5] = {1,2,3,4,5};
        float B[5] = {60,2,3,4,5};
        clEnqueueWriteBuffer(command_queue,A_buf,CL_TRUE,0,sizeof(float)*5,A,0,NULL,NULL);
        clEnqueueWriteBuffer(command_queue,B_buf,CL_TRUE,0,sizeof(float)*5,B,0,NULL,NULL);

        // 启动kernel
        size_t gSize[3] = {5,1,1};
        size_t lSize[3] = {5,1,1};
        status = clEnqueueNDRangeKernel(command_queue,kernel,1,NULL,gSize,lSize,0,NULL,NULL);
        checkError(status,"Failed to launch kernel");

        // 设备传回数据
        float C[5];
        clEnqueueReadBuffer(command_queue,C_buf,CL_TRUE,0,sizeof(float)*5,C,0,NULL,NULL);
        
        // 完成
        status = clFinish(command_queue);
        checkError(status,"Failed to Finish");

        // 打印C数据数据
        for(int i=0;i<5;++i){
            printf("C[%d]=%f ",i,C[i]);
        }

        printf("/n");
        
    }

    void learn(vector<double> x,double y){

        numSamplesTrained += 1;

        printf("\n############%d################\n",numSamplesTrained);

        X.push_back(x);
        Y.push_back(y);
        weights.push_back(0);

        vector<double> H = computeMargin(X,Y);
        if(abs(H[numSamplesTrained-1])  <= eps ){
            printf("第%d个样本加入R集合，由于eps\n",numSamplesTrained);
            remainderSetIndices.push_back(numSamplesTrained-1);
            return ;
        }

        bool newSampleAdded = false;
        int iterations = 0;

        while(!newSampleAdded){
            ++iterations;

           if(iterations > numSamplesTrained*100){
                printf("迭代次数过多\n");
                return ;
            }

            vector<vector<double>> beta,gamma;
            computeBetaGamma(numSamplesTrained-1,beta,gamma);

            double deltaC;
            int flag;
            int minIndex;
            findMinVariation(H,beta,gamma,numSamplesTrained-1,deltaC,flag,minIndex);

            if(supportSetIndices.size() > 0 && beta.size() > 0){
                weights[numSamplesTrained-1] += deltaC;
                vector<double> delta;
                for(int j=0;j<beta.size();++j){
                    delta.push_back(beta[j][0] * deltaC);
                }

                bias += delta[0];

                for(int j=0;j<supportSetIndices.size();++j){
                    weights[supportSetIndices[j]] += delta[j+1];
                }

                for(int j=0;j<H.size();++j){
                    H[j] += gamma[j][0] * deltaC;
                }
                vector<double> H2 = computeMargin(X,Y);
                int testtt = 3;

            }else{
                bias += deltaC;
                for(int j=0;j<H.size();++j){
                    H[j] += deltaC;
                }
            }
            
            adjustSets(H,beta,gamma,numSamplesTrained-1,flag,minIndex,newSampleAdded);

        }  

        printf("iterations=%d\n",iterations);

    }

    void adjustSets(vector<double> H,vector<vector<double>> beta,vector<vector<double>> gamma,int i,int flag,int minIndex
    ,bool & newSampleAdded){
        if(flag >= 5){
            return ;
        }

        if(flag == 0){
            int sign = (H[i]>0)?1:(-1);
            H[i] = sign * eps;
            supportSetIndices.push_back(i);
            R = addSampleToR(i,"SupportSet",beta,gamma);
            newSampleAdded = true;
            return;
        }else if(flag == 1){
            weights[i] = sign(weights[i])*C;
            errorSetIndices.push_back(i);
            newSampleAdded = true;
            return ;
        }else if(flag == 2){
            int index = supportSetIndices[minIndex];
            double weightsValue = weights[index];
            if(abs(weightsValue) < abs(C - abs(weightsValue))){
                weights[index] = 0;
                weightsValue = 0;
            }else {
                weights[index] = sign(weightsValue) * C;
                weightsValue = weights[index];
            }

            if(weightsValue == 0){
                remainderSetIndices.push_back(index);
                R = removeSampleFromR(minIndex);
                supportSetIndices.erase(supportSetIndices.begin() + minIndex);
            }else if(abs(weightsValue) == C){
                errorSetIndices.push_back(index);
                R = removeSampleFromR(minIndex);
                supportSetIndices.erase(supportSetIndices.begin() + minIndex);
            }


        }else if(flag == 3){
            int index = errorSetIndices[minIndex];
            H[index] = sign(H[index]) * eps;
            supportSetIndices.push_back(index);
            errorSetIndices.erase(errorSetIndices.begin() + minIndex);
           
            R = addSampleToR(index,"ErrorSet",beta,gamma);

        }else if(flag == 4){
            int index = remainderSetIndices[minIndex];
            H[index] = sign(H[index]) * eps;
            supportSetIndices.push_back(index);
            remainderSetIndices.erase(remainderSetIndices.begin() + minIndex);
            R = addSampleToR(index,"RemainingSet",beta,gamma);
        }

    }

    int sign(double num){
        return (num>=0)?1:(-1);
    }


    vector<vector<double>> addSampleToR(int sampleIndex,string sampleOldSet ,vector<vector<double>> beta,vector<vector<double>> gamma){
        vector<vector<double>> sampleX(1,X[sampleIndex]);
        vector<vector<double>> Rnew;
        if(R.size() <= 1){
            Rnew.push_back(vector<double>(2,1));
            Rnew.push_back(vector<double>(2,1));
            auto Qxx = computeKernelOutput(sampleX,sampleX);
            Rnew[0][0] = -Qxx[0][0];
            Rnew[1][1] = 0;
        }else{
            if(sampleOldSet == "ErrorSet" || sampleOldSet=="RemainingSet"){
                vector<vector<double>> Qii = computeKernelOutput(sampleX,sampleX);
                vector<vector<double>> sVec;
                for(int j=0;j<supportSetIndices.size();++j){
                    sVec.push_back(X[supportSetIndices[j]]);
                }
                vector<vector<double>> Qsi = computeKernelOutput(sVec,sampleX);
                Qsi.insert(Qsi.begin(),vector<double>(1,1));;
                vector<vector<double>> minusR;
                for(int j=0;j<R.size();++j){
                    vector<double> curRow;
                    for(int k=0;k<R[j].size();++k){
                        curRow.push_back(-R[j][k]);
                    }

                    minusR.push_back(curRow);
                }

                beta = matMul(minusR,Qsi);
                
                gamma[sampleIndex][0] = Qii[0][0];
                for(int j=0;j<beta.size();++j){
                    gamma[sampleIndex][0] += Qsi[j][0]*beta[j][0]; 
                }
            }

            for(auto &rows:R){
                rows.push_back(0);
            }
            int colNum = R[0].size();
            R.push_back(vector<double>(colNum,0));

            if(gamma[sampleIndex][0] != 0){
            
                vector<vector<double>> beta1(beta);
                beta1.push_back(vector<double>(1,1));
                vector<vector<double>> betaT = matT(beta1);
                double num = 1.0/gamma[sampleIndex][0];
                vector<vector<double>> mulResult = matMulNum(matMul(beta1,betaT),num);
                
                Rnew = matAdd(R,mulResult);
            }
        }

        return Rnew;

    }

    vector<vector<double>> removeSampleFromR(int sampleIndex){
        vector<vector<double>> results;

        sampleIndex += 1;
        vector<int> I;
        for(int j=0;j<sampleIndex;++j){
            I.push_back(j);
        }

        for(int j=sampleIndex+1;j<R.size();++j){
            I.push_back(j);
        }

        vector<vector<double>> R_I_I;
            for(auto j:I){
                vector<double> curRow;
                for(auto k:I){
                    curRow.push_back(R[j][k]);
                }
                R_I_I.push_back(curRow);
            }

        if(R[sampleIndex][sampleIndex] != 0){
            

            vector<vector<double>> R_IT_I;
            for(auto j:I){
                vector<double> curRow;
                for(auto k:I){
                    curRow.push_back(R[j][sampleIndex] * R[sampleIndex][k]);
                }
                R_IT_I.push_back(curRow);
            }

            results = matMinus(R_I_I,matMulNum(R_IT_I,1.0/(R[sampleIndex][sampleIndex])));

        }else{
            results = R_I_I;
        }

        return results;
    }

    vector<vector<double>> matMinus(const vector<vector<double>> & v1,const vector<vector<double>> &v2){
        vector<vector<double>> results;

        for(int j=0;j<v1.size();++j){
            vector<double> curRow;
            for(int k=0;k<v1[0].size();++k){
                curRow.push_back(v1[j][k]-v2[j][k]);
            }
            results.push_back(curRow);
        }

        return results;
    }

    vector<vector<double>> matMulNum(const vector<vector<double>> &v,double num){
        vector<vector<double>> results;
        for(int j=0;j<v.size();++j){
            vector<double> curRow;
            for(int k=0;k<v[0].size();++k){
                curRow.push_back(v[j][k] * num);
            }
            results.push_back(curRow);
        }
        return results;
    }

    vector<vector<double>> matAdd(const vector<vector<double>> &v1,const vector<vector<double>> &v2){
        vector<vector<double>> results;
        for(int j=0;j<v1.size();++j){
            vector<double> curRow;
            for(int k=0;k<v1[0].size();++k){
                curRow.push_back(v1[j][k] + v2[j][k]);
            }
            results.push_back(curRow);
        }

        return results;

    }

    vector<vector<double>> matT(const vector<vector<double>> &v){
        vector<vector<double>> results;
        if(v.size()==0){
            return results;
        }

        for(int j=0;j<v[0].size();++j){
            vector<double> curRow;
            for(int k=0;k<v.size();++k){
                curRow.push_back(v[k][j]);
            }
            results.push_back(curRow);
        }

        return results;
    }

 

    void findMinVariation(vector<double> H,vector<vector<double>> beta,vector<vector<double>> gamma,int i,
    double &deltaC,int &flag,int &minIndex){
        int q = (H[i])>=0?-1:1;



        double Lc1 = findVarLc1(H,gamma,q,i);

        q = (Lc1>=0)?1:(-1);
        
        double Lc2 = findVarLc2(H,q,i);

        vector<double> Ls = findVarLs(H,beta,q);
        vector<double> Le = findVarLe(H,gamma,q);
        vector<double> Lr = findVarLr(H,gamma,q);

        if(Ls.size() > 1){
            double minS = INFINITY;
            for(auto lsValue:Ls){
                if(abs(lsValue)<minS){
                    minS = abs(lsValue);
                }
            }

            vector<int> results;
            for(int j=0;j<Ls.size();++j){
                if(abs(Ls[j]) == minS){
                    results.push_back(j);
                } 
            }

            if(results.size() > 1) {
                double max = -INFINITY;
                int maxIdx = -1;
                for(int j=0;j<results.size();++j){
                    if(beta[results[j]+1][0] > max){
                        max = beta[results[j]+1][0];
                        maxIdx = results[j] + 1;
                    }

                    Ls[results[j]] = q*INFINITY;
                    
                }

                Ls[results[maxIdx]] = q*minS;
            }
        }
        if(Le.size() > 1){
            double minE = INFINITY;
            for(int j=0;j<Le.size();++j){
                if(abs(Le[j]) < minE){
                    minE = Le[j];
                }
            }

            vector<int> results;

            for(int j=0;j<Le.size();++j){
                if(abs(Le[j] == minE)){
                    results.push_back(j);
                }
            }

            if(results.size() > 1){
                vector<double> errorGamma ;
                for(int j=0;j<errorSetIndices.size();++j){
                    errorGamma.push_back(gamma[errorSetIndices[j]][0]);
                }
                double maxErrorGamma = -INFINITY;
                int gammaIndex = -1;
                for(int j=0;j<results.size();++j){
                    if(errorGamma[results[j]] >maxErrorGamma){
                        maxErrorGamma = errorGamma[results[j]];
                        gammaIndex = j;
                    }

                    Le[results[j]] = q*INFINITY;
                }

                Le[results[gammaIndex]] = q*minE;
            }


        }
        if(Lr.size()){
            double minR = INFINITY;
            
            for(int j=0;j<Lr.size();++j){
                if(abs(Lr[j]) < minR){
                    minR = abs(Lr[j]);
                }
            }

            vector<int> results;

            for(int j=0;j<Lr.size();++j){
                if(abs(Lr[j])  ==minR ){
                    results.push_back(j);
                }
            }

            if(results.size() > 1){

                vector<double> remGamma;
                for(int j=0;j<remainderSetIndices.size(); ++j){
                    remGamma.push_back(gamma[remainderSetIndices[j]][0]);
                }


                double maxValue = -INFINITY;
                int maxIdx = -1;
                for(int j=0;j<results.size();++j){
                    if(remGamma[results[j]] > maxValue){
                        maxValue = remGamma[results[j]];
                        maxIdx = j;
                    }
                    Lr[results[j]] = q*INFINITY;
                }

                Lr[results[maxIdx]] = q*minR;

            }
        }

        int minLsIndex = min_element(Ls.begin(),Ls.end(),[](const double &m1,const double & m2){return abs(m1)<abs(m2);}) - Ls.begin();
        int minLeIndex = min_element(Le.begin(),Le.end(),[](const double &m1,const double & m2){return abs(m1)<abs(m2);}) - Le.begin();
        int minLrIndex = min_element(Lr.begin(),Lr.end(),[](const double &m1,const double & m2){return abs(m1)<abs(m2);}) - Lr.begin();
        vector<int> minIndices = {-1,-1,minLsIndex,minLeIndex,minLrIndex};
        vector<double> minValues = {Lc1,Lc2,Ls[minLsIndex],Le[minLeIndex],Lr[minLrIndex]};
        
        flag = min_element(minValues.begin(),minValues.end(),[](const auto &m1,const auto & m2){return abs(m1)<abs(m2);}) - minValues.begin();
        deltaC = minValues[flag];
        minIndex = minIndices[flag];
        //printf(" ");
    }

    double findVarLc1(vector<double> H,vector<vector<double>> gamma,int q,int i){
        double Lc1 = 0;
        double gammaValue = gamma[i][0];

        if(gammaValue<0){
            Lc1 = q*INFINITY;
        }else if(H[i]>eps && weights[i] >(-C) && weights[i]<= 0){
            Lc1 = (-H[i] + eps) / gammaValue;
        }else if(H[i]<(-eps) && weights[i]>=0 && weights[i] <= C ){
            Lc1 = (-H[i] - eps) / gammaValue;
        }

        return Lc1;

    }

    double findVarLc2(vector<double> H,int q,int i){
        double Lc2 = 0;

        if(supportSetIndices.size() >0){
            Lc2 = -weights[i] + q*C;
        }else{
            Lc2 = q*INFINITY;
        }
        return Lc2;
    }

    vector<double>  findVarLs(vector<double> H,vector<vector<double>> beta,int q){
        vector<double> Ls;

        if(supportSetIndices.size() > 0 && beta.size() > 0){
            vector<double> supportWeigths;
            vector<double> supportH;
            
            for(int i=0;i<supportSetIndices.size();++i){
                Ls.push_back(0);
                supportWeigths.push_back(weights[supportSetIndices[i]]);
                supportH.push_back(H[supportSetIndices[i]]);
            }

            for(int i=0;i<supportSetIndices.size();++i){
                if(q*beta[i+1][0] == 0){
                    Ls[i] = q*INFINITY; 
                }else if(q*beta[i+1][0] >0 ){
                    if(supportH[i]>0){
                        if(supportWeigths[i]< -C){
                            Ls[i] = (-supportWeigths[i]-C)/beta[i+1][0];
                        }else if(supportWeigths[i] <=0){
                            Ls[i] = -supportWeigths[i]/beta[i+1][0];
                        }else{
                            Ls[i] = q*INFINITY;
                        }
                    }else{
                        if(supportWeigths[i] < 0){
                            Ls[i] = -supportWeigths[i]/beta[i+1][0];
                        }else if (supportWeigths[i] <= C){
                            Ls[i] = (-supportWeigths[i] + C)/beta[i+1][0];
                        }else {
                            Ls[i] = q*INFINITY;
                        }
                    }
                }else{
                    if(supportH[i] > 0){
                        if(supportWeigths[i] > 0){
                            Ls[i] = -supportWeigths[i] / beta[i+1][0];
                        }else if(supportWeigths[i] >= -C){
                            Ls[i] = (-supportWeigths[i] - C) / beta[i+1][0];
                        }else {
                            Ls[i] = q*INFINITY;
                        }
                    }else{
                        if(supportWeigths[i] > C){
                            Ls[i] = (-supportWeigths[i] + C)/beta[i+1][0];
                        }else if(supportWeigths[i] >= C){
                            Ls[i] = -supportWeigths[i] / beta[i+1][0];
                        }else { 
                            Ls[i] = q*INFINITY;
                        }
                    }
                }

            }
            


        }else{
            Ls.push_back(q*INFINITY);
        }


        return Ls;
    }


    vector<double> findVarLe(vector<double> H,vector<vector<double>> gamma,int q){
        vector<double> Le;
        vector<double> errorGamma;
        vector<double> errorWeights;
        vector<double> errorH;

        if(errorSetIndices.size() > 0){
            for(int i=0;i<errorSetIndices.size();++i){
                Le.push_back(0);
                errorWeights.push_back(weights[errorSetIndices[i]]);
                errorH.push_back(H[errorSetIndices[i]]);
                errorGamma.push_back(gamma[errorSetIndices[i]][0]);
            }

            for(int i=0;i<errorSetIndices.size();++i){
                if(q*errorGamma[i]  ==0){
                    Le[i] = q*INFINITY; 
                }else if(q*errorGamma[i] >0){
                    if(errorWeights[i] >0 ){
                        if(errorH[i] < -eps){
                            Le[i] = (-errorH[i] - eps) / errorGamma[i];
                        }else{
                            Le[i] = q*INFINITY;
                        }
                    }else{
                        if(errorH[i] < eps){
                            Le[i] = (-errorH[i] + eps) /errorGamma[i];
                        }else{
                            Le[i] = q*INFINITY;
                        }
                    }

                }else{
                    if(errorWeights[i] > 0){
                        if(errorH[i] > -eps){
                            Le[i] = (-errorH[i] - eps) / errorGamma[i];
                        }else{
                            Le[i] = q*INFINITY;
                        }
                    }else{
                        if(errorH[i] > eps){
                            Le[i] = (-errorH[i] + eps) / errorGamma[i];
                        }else{
                            Le[i] = q*INFINITY;
                        }

                    }
                    
                }
            }


        }else{
            Le.push_back(q*INFINITY);
        }

        return Le;
    }

    vector<double> findVarLr(vector<double> H,vector<vector<double>> gamma,int q){
        vector<double> Lr;
        vector<double> remGamma;
        vector<double> remH;

        if(remainderSetIndices.size() > 0){
            for(int i=0;i<remainderSetIndices.size();++i){
                Lr.push_back(0);
                remGamma.push_back(gamma[remainderSetIndices[i]][0]);
                remH.push_back(H[remainderSetIndices[i]]);
            }    

            for(int i=0;i<remainderSetIndices.size();++i){
                if(q*remGamma[i] ==0 ){
                    Lr[i] = q*INFINITY;
                }else if(q*remGamma[i] > 0){
                    if(remH[i] < -eps){
                        Lr[i] = (-remH[i] - eps) / remGamma[i];
                    }else if(remH[i] < eps  ){
                        Lr[i] = (-remH[i] + eps ) / remGamma[i];
                    }else{
                        Lr[i] = q*INFINITY;
                    }
                }else{
                    if(remH[i] > eps){
                        Lr[i] = (-remH[i] + eps)/remGamma[i];
                    }else if(remH[i] > -eps){
                        Lr[i] = (-remH[i] - eps)/remGamma[i];
                    }else{
                        Lr[i] = q*INFINITY;
                    }
                }
            }

        
        }else{
            Lr.push_back(q*INFINITY);
        }



        return Lr;
    }

    void computeBetaGamma(int i,vector<vector<double>> &beta,vector<vector<double>> &gamma){

        vector<vector<double>> sVec;
        for(int j=0;j<supportSetIndices.size();++j){
            sVec.push_back(X[supportSetIndices[j]]);
        }

    
        vector<vector<double>> cVec(1,X[i]);


        vector<vector<double>> Qsc = computeKernelOutput(sVec,cVec);

        Qsc.insert(Qsc.begin(),vector<double>(1,1));

        if (! (supportSetIndices.size() == 0 || R.size() == 0 )){
            // 
            vector<vector<double>> minusR;
            for(auto row:R){
                vector<double> curRow;
                for(auto ele:row){
                    curRow.push_back(-ele);
                }
                minusR.push_back(curRow);
            }
            beta = matMul(minusR,Qsc);
        }else{
            vector<vector<double>> temp;
            beta = temp;
        }
        
        vector<vector<double>> Qxc = computeKernelOutput(X,cVec);
        vector<vector<double>> Qxs = computeKernelOutput(X,sVec);
        if(supportSetIndices.size() ==0 || Qxc.size() == 0 || Qxs.size() == 0 || beta.size()==0){
            for(auto row:Qxc){
                vector<double> curRow;
                for(auto ele:row){
                    curRow.push_back(1);
                }
                gamma.push_back(curRow);
            }
        }else{
            for(auto &row:Qxs){
                row.insert(row.begin(),1);
            }
            auto multiplication = matMul(Qxs,beta);
            auto iter = gamma.begin();
            while(iter != gamma.end()){
                iter = gamma.erase(iter);
            }
            for(int j = 0;j<Qxc.size();++j){
                vector<double> row;
                row.push_back(Qxc[j][0] + multiplication[j][0]);
                gamma.push_back(row);
            }

            
          //  printf("ceshi");

        }


    }

    vector<vector<double>> matMul(vector<vector<double>> mat1,vector<vector<double>> mat2){
        vector<vector<double>> results;

        for(int i=0;i<mat1.size();++i){
            vector<double> row;

            for(int j=0;j<mat2[0].size();++j){
                double sum=0;

                for(int k=0;k<mat1[i].size();++k){
                    sum += mat1[i][k] * mat2[k][j];
                }

                row.push_back(sum);
            }


            results.push_back(row);

        }

        return results;
    }

    vector<double> computeMargin(vector<vector<double>> newX,vector<double> newY){

        vector<double> fx = predict(newX);

        for(int i=0;i<fx.size();++i){
            fx[i] -= newY[i];
        }

        return fx;
    }   

    vector<double> predict(vector<vector<double>> newX){
        vector<double> results;

        vector<vector<double>> v = computeKernelOutput(X,newX);
        
        for(int i=0;i<newX.size();++i){
            double result = 0;

            for(int j=0;j<weights.size();++j){
                result += weights[j]*v[j][i];
            }

            result += bias;
            
            results.push_back(result);
        }
        


        return results;
    }

    /*
    vector<vector<double>> computeKernelOutput(vector<vector<double>> set1,vector<vector<double>> set2){
        vector<vector<double>> result;
        for(int i=0;i<set1.size();++i){
            vector<double> curV;
            for(int j=0;j<set2.size();++j){
                double rbfValue = 0;
                vector<double> v1 = set1[i];
                vector<double> v2 = set2[j];

                rbfValue = rbf(v1,v2);

                curV.push_back(rbfValue);
            }

            result.push_back(curV);
        }

        return result;
    }
    */

    
    // FPGA
   vector<vector<double>> computeKernelOutput(vector<vector<double>> set1,vector<vector<double>> set2){
        


    }
    

    double rbf(vector<double> v1,vector<double> v2){
        double result =0;
        
        double squareSum = 0;
        for(int i=0;i<v1.size();++i){
            squareSum += (v1[i]-v2[i]) * (v1[i]-v2[i]); 
        }
        result = exp(squareSum*(-kernelParam));

        return result;
    }   


private:
    int numFeatures;
    int C;
    double eps;
    double kernelParam;
    double bias;

    int numSamplesTrained ;

    vector<double> weights;
    vector<vector<double>> X;
    vector<double> Y;
    
    vector<int> supportSetIndices;
    vector<int> errorSetIndices;
    vector<int> remainderSetIndices;

    vector<vector<double>> R;

    cl_command_queue command_queue;
    cl_kernel kernel;
    cl_mem A_buf;
    cl_mem B_buf;
    cl_mem C_buf;

    

};


#endif

