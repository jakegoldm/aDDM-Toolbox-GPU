#include <cuda.h>
#include <cuda_runtime.h>
#include <cassert>
#include "addm.h"
#include "ddm.h"
#include "cuda_util.cuh"
#include "util.h"


__global__
void getTrialLikelihoodKernel(
    bool debug, 
    int trialsPerThread, 
    int *RTs, 
    int *choices, 
    int *valLs, 
    int *valRs, 
    int *FixItemsMatrix, 
    int *FixTimeMatrix, 
    int *FixLens, 
    double *likelihoods, 
    int numTrials, 
    float *states, 
    int maxFixLen, 
    int biasState, 
    int numStates,
    float stateStep,
    float d, 
    float sigma, 
    float theta, 
    int barrier, 
    int nonDecisionTime, 
    int timeStep, 
    float approxStateStep, 
    float dec,
    double *prStates, 
    double* prStatesNew) {

    int tid = blockIdx.x * blockDim.x + threadIdx.x; 
    if (tid < numTrials / trialsPerThread) {
        float *probDistChangeMatrix = new float[numStates * numStates];
        float *changeUpCDFs = new float[numStates];
        float *changeDownCDFs = new float[numStates];
        float *changeMatrix = new float[numStates * numStates];
        for (int trialNum = tid * trialsPerThread; trialNum < (tid + 1) * trialsPerThread; trialNum++) {
            
            int choice = choices[trialNum];
            int RT = RTs[trialNum];
            int valLeft = valLs[trialNum];
            int valRight = valRs[trialNum];
            int fixLen = FixLens[trialNum];

            int *fixItem = new int[fixLen];
            int *fixTime = new int[fixLen];

            int f_idx = 0; 
            #pragma unroll 4
            for (int i = trialNum * maxFixLen; i < trialNum * maxFixLen + fixLen; i++) {
                fixItem[f_idx] = FixItemsMatrix[i];
                fixTime[f_idx] = FixTimeMatrix[i];
                f_idx++; 
            }

            if (debug) {
                printf("%i %i %i %i\n", choice, RT, valLeft - valRight, fixLen);
                printf("Fix Item | Fix Time \n");
                for (int i = 0; i < fixLen; i++) {
                    printf("%i        | %i   \n", fixItem[i], fixTime[i]);
                } 
            }

            int numTimeSteps = 0; 
            for (int i = 0; i < fixLen; i++) {
                numTimeSteps += fixTime[i] / timeStep; 
            }
            numTimeSteps++; 

            // requires compute capability 2.x
            float* barrierUp = new float[numTimeSteps];
            float *barrierDown = new float[numTimeSteps];

            #pragma unroll 4
            for (int i = 0 ; i < numTimeSteps; i++) {
                barrierUp[i] = barrier / (1 + (dec * i));
                barrierDown[i] = -barrier / (1 + (dec * i));
            }

            #pragma unroll 4
            for (int i = 0; i < numStates; i++) {
                prStates[__RC2IDX(trialNum, i, numStates)] = (i == biasState) ? 1 : 0; 
            }

            double *probUpCrossing = new double[numTimeSteps];
            double *probDownCrossing = new double[numTimeSteps];
            #pragma unroll 4
            for (int i = 0; i < numTimeSteps; i++) {
                probUpCrossing[i] = 0; 
                probDownCrossing[i] = 0; 
            }

            if (debug) {
                for (int i = 0 ; i < numStates ; i++) {
                    printf("prStates[%i] = %f\n", i, prStates[__RC2IDX(trialNum, i, numStates)]);
                }
            }

            int time = 1;
            #pragma unroll 4
            for (int i = 0; i < numStates; i++) {
                #pragma unroll 4
                for (int j = 0; j < numStates; j++) {
                    changeMatrix[__RC2IDX(i, j, numStates)] = states[i] - states[j];
                }
            }

            float *changeUp = new float[numStates * numTimeSteps];
            #pragma unroll 4
            for (int i = 0; i < numStates; i++) {
                #pragma unroll 4
                for (int j = 0; j < numTimeSteps; j++) {
                    changeUp[__RC2IDX(i, j, numTimeSteps)] = barrierUp[j] - states[i];
                }
            }

            float *changeDown = new float[numStates * numTimeSteps];
            #pragma unroll 4
            for (int i = 0; i < numStates; i++) {
                #pragma unroll 4
                for (int j = 0; j < numTimeSteps; j++) {
                    changeDown[__RC2IDX(i, j, numTimeSteps)] = barrierDown[j] - states[i];
                }
            }

            for (int f = 0; f < fixLen; f++) {
                int fItem = fixItem[f];
                int fTime = fixTime[f];

                if (debug) {
                    printf("fItem : %i ========== fTime : %i\n", fItem, fTime);
                }

                float mean; 
                if (fItem == 1) {
                    mean = d * (valLeft - (theta * valRight));
                } else if (fItem == 2) {
                    mean = d * ((theta * valLeft) - valRight);
                } else {
                    mean = 0; 
                }

                #pragma unroll 4
                for (int i = 0; i < numStates; i++) {
                    #pragma unroll 4
                    for (int j = 0; j < numStates; j++) {
                        float x = changeMatrix[__RC2IDX(i, j, numStates)];
                        probDistChangeMatrix[__RC2IDX(i, j, numStates)] = __pdf(x, mean, sigma);
                    }
                }
                if (debug) {
                    printf("PDCM\n");
                    for (int i = 0; i < numStates * numStates; i++) {
                        printf("%f ", probDistChangeMatrix[i]);
                        if ((i + 1) % numStates == 0) {
                            printf("\n");
                        }
                    }
                }

                double tempUpCross; 
                double tempDownCross; 
                if (dec == 0) {
                    #pragma unroll 4 
                    for (int i = 0; i < numStates; i++) {
                        float x = changeUp[__RC2IDX(i, time, numTimeSteps)];
                        changeUpCDFs[i] = 1 - normcdff((x - mean) / sigma);
                    }    
                    #pragma unroll 4
                    for (int i = 0; i < numStates; i++) {
                        float x = changeDown[__RC2IDX(i, time, numTimeSteps)];
                        changeDownCDFs[i] = normcdff((x - mean) / sigma);
                    } 
                }

                for (int t = 0; t < fTime / timeStep; t++) {
                    double rowSum; 
                    for (int i = 0; i < numStates; i++) {
                        rowSum = 0; 
                        for (int j = 0; j < numStates; j++) {
                            rowSum += stateStep * probDistChangeMatrix[__RC2IDX(i, j, numStates)] * prStates[__RC2IDX(trialNum, j, numStates)];
                        }
                        prStatesNew[__RC2IDX(trialNum, i, numStates)] = (states[i] > barrierUp[time] || states[i] < barrierDown[time]) ? 0 : rowSum;
                    }

                    if (debug) {
                        for (int i = 0 ; i < numStates ; i++) {
                            printf("prStatesNew[%i] = %f\n", i, prStatesNew[__RC2IDX(trialNum, i, numStates)]);
                        }
                    }

                    if (dec != 0) {
                        #pragma unroll 4 
                        for (int i = 0; i < numStates; i++) {
                            float x = changeUp[__RC2IDX(i, time, numTimeSteps)];
                            changeUpCDFs[i] = 1 - normcdff((x - mean) / sigma);
                        }
                        if (debug) {
                            for (int i = 0; i < numStates; i++) {
                                printf("changeUpCDFs[%i] = %f\n", i, changeUpCDFs[i]);
                            }
                        }
                        #pragma unroll 4                    
                        for (int i = 0; i < numStates; i++) {
                            float x = changeDown[__RC2IDX(i, time, numTimeSteps)];
                            changeDownCDFs[i] = normcdff((x - mean) / sigma);
                        }
                        if (debug) {
                            for (int i = 0; i < numStates; i++) {
                                printf("changeDownCDFs[%i] = %f\n", i, changeDownCDFs[i]);
                            }
                        }                  
                    }

                    tempUpCross = 0; 
                    for (int i = 0; i < numStates; i++) {
                        tempUpCross += changeUpCDFs[i] * prStates[__RC2IDX(trialNum, i, numStates)];
                    }     
                    tempDownCross = 0; 
                    for (int i = 0; i < numStates; i++) {
                        tempDownCross += changeDownCDFs[i] * prStates[__RC2IDX(trialNum, i, numStates)];
                    }     

                    double sumIn = 0; 
                    double sumCurrent = tempUpCross + tempDownCross; 

                    #pragma unroll 4
                    for (int i = 0; i < numStates; i++) {
                        sumIn += prStates[__RC2IDX(trialNum, i, numStates)];
                        sumCurrent += prStatesNew[__RC2IDX(trialNum, i, numStates)];
                    }
                    double normFactor = sumIn / sumCurrent; 

                    #pragma unroll 4
                    for (int i = 0; i < numStates; i++) {
                        prStates[__RC2IDX(trialNum, i, numStates)] = prStatesNew[__RC2IDX(trialNum, i, numStates)] * normFactor; 
                    }

                    probUpCrossing[time] = tempUpCross * normFactor; 
                    probDownCrossing[time] = tempDownCross * normFactor;

                    time++;
                }
            }

            double likelihood = 0; 
            if (choice == -1) {
                if (probUpCrossing[numTimeSteps - 1] > 0) {
                    likelihood = probUpCrossing[numTimeSteps - 1];
                }
            } else if (choice == 1) {
                if (probDownCrossing[numTimeSteps - 1] > 0) {
                    likelihood = probDownCrossing[numTimeSteps - 1];
                }
            }

            delete[] fixItem; 
            delete[] fixTime; 
            delete[] barrierUp;
            delete[] barrierDown;
            delete[] probUpCrossing;
            delete[] probDownCrossing;
            delete[] changeUp;
            delete[] changeDown;

            if (likelihood == 0) {
                likelihood = pow(10, -20);
            }   
            likelihoods[trialNum] = likelihood;            
        }
 
        delete[] changeMatrix; 
        delete[] probDistChangeMatrix;
        delete[] changeUpCDFs;
        delete[] changeDownCDFs;
    }
}


void aDDM::callGetTrialLikelihoodKernel(
    bool debug, 
    int trialsPerThread,
    int numBlocks,
    int threadsPerBlock, 
    aDDMTrial *trials, 
    double *likelihoods, 
    int numTrials, 
    float d, 
    float sigma, 
    float theta, 
    float barrier, 
    int nonDecisionTime, 
    int timeStep, 
    float approxStateStep, 
    float decay
) {
    int *h_fixLens = new int[numTrials];
    int maxFixLen = 0; 
    for (int i = 0; i < numTrials; i++) {
        aDDMTrial trial = trials[i];
        assert(trial.fixItem.size() == trial.fixTime.size());
        int fixLen = trial.fixItem.size();
        if (fixLen > maxFixLen) {
            maxFixLen = fixLen;
        }
        h_fixLens[i] = fixLen; 
    }

    if (debug) std::cout << "max fix len " << maxFixLen << std::endl; 


    int *h_FIs = new int[numTrials * maxFixLen];
    int *h_FTs = new int[numTrials * maxFixLen];
    for (int i = 0; i < numTrials; i++) {
        for (int j = 0; j < maxFixLen; j++) {
            aDDMTrial trial = trials[i];
            if (j < trial.fixItem.size()) {
                h_FIs[__RC2IDX(i, j, maxFixLen)] = trial.fixItem[j]; 
                h_FTs[__RC2IDX(i, j, maxFixLen)] = trial.fixTime[j]; 
            } else {
                h_FIs[__RC2IDX(i, j, maxFixLen)] = -1; 
                h_FTs[__RC2IDX(i, j, maxFixLen)] = -1; 
            }
        }
    }

    if (debug) {
        for (int i = 0; i < numTrials; i++) {
            std::cout << "[" << h_fixLens[i] << "] ";
            for (int j = 0; j < maxFixLen; j++) {
                std::cout << h_FIs[__RC2IDX(i, j, maxFixLen)] << " ";  
            }
            std::cout << std::endl;
        }
    }

    int *h_VLs = new int[numTrials];
    int *h_VRs = new int[numTrials];
    int *h_RTs = new int[numTrials];
    int *h_choices = new int[numTrials];
    for (int i = 0; i < numTrials; i++) {
        aDDMTrial trial = trials[i];
        h_VLs[i] = trial.valueLeft; 
        h_VRs[i] = trial.valueRight; 
        h_RTs[i] = trial.RT;
        h_choices[i] = trial.choice; 
    }

    int *d_RTs, *d_choices, *d_VLs, *d_VRs, *d_FIs, *d_FTs, *d_FixLens;
    cudaMalloc((void **) &d_RTs, numTrials * sizeof(int));
    cudaMalloc((void **) &d_choices, numTrials * sizeof(int));
    cudaMalloc((void **) &d_VLs, numTrials * sizeof(int));
    cudaMalloc((void **) &d_VRs, numTrials * sizeof(int));
    cudaMalloc((void **) &d_FIs, numTrials * maxFixLen * sizeof(int));
    cudaMalloc((void **) &d_FTs, numTrials * maxFixLen * sizeof(int)); 
    cudaMalloc((void **) &d_FixLens, numTrials * sizeof(int));

    cudaMemcpy(d_RTs, h_RTs, numTrials * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_choices, h_choices, numTrials * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_VLs, h_VLs, numTrials * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_VRs, h_VRs, numTrials * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_FIs, h_FIs, numTrials * maxFixLen * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_FTs, h_FTs, numTrials * maxFixLen * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_FixLens, h_fixLens, numTrials * sizeof(int), cudaMemcpyHostToDevice);

    int halfNumStateBins = ceil(barrier / approxStateStep); 
    if (debug) printf("half num state bins %i\n", halfNumStateBins);
    float stateStep = barrier / (halfNumStateBins + 0.5);
    if (debug) printf("state step %f\n", stateStep);
    int numStates = 2 * halfNumStateBins + 1; 

    float *states = new float[numStates];
    int s_idx = 0; 
    float biasStateVal = MAXFLOAT; 
    int biasState; 
    float r; 
    
    for (float ss = -barrier + (stateStep / 2); ss <= barrier - (stateStep / 2); ss += stateStep) {
        states[s_idx] = ss;
        r = abs(ss - bias); 
        if (r < biasStateVal) {
            biasState = s_idx;
            biasStateVal = r; 
        }
        s_idx++;
    }

    float *d_states; 
    cudaMalloc((void**) &d_states, numStates * sizeof(float));
    cudaMemcpy(d_states, states, numStates * sizeof(float), cudaMemcpyHostToDevice);

    double *d_prStates, *d_prStatesNew; 
    cudaMalloc((void**) &d_prStates, numStates * numTrials * sizeof(double));
    cudaMalloc((void**) &d_prStatesNew, numStates * numTrials * sizeof(double));

    getTrialLikelihoodKernel<<<numBlocks, threadsPerBlock>>>(
        debug, 
        trialsPerThread, 
        d_RTs, 
        d_choices, 
        d_VLs, 
        d_VRs, 
        d_FIs, 
        d_FTs, 
        d_FixLens, 
        likelihoods, 
        numTrials, 
        d_states, 
        maxFixLen,
        biasState, 
        numStates,
        stateStep,
        d, sigma, theta, barrier, 
        nonDecisionTime, 
        timeStep, 
        approxStateStep, 
        decay,
        d_prStates, 
        d_prStatesNew
    );

    cudaFree(d_RTs);
    cudaFree(d_choices);
    cudaFree(d_VLs);
    cudaFree(d_VRs);
    cudaFree(d_FIs);
    cudaFree(d_FTs);
    cudaFree(d_FixLens);
    cudaFree(d_prStates);
    cudaFree(d_prStatesNew);
    delete[] h_RTs;
    delete[] h_choices; 
    delete[] h_VLs;
    delete[] h_VRs;
    delete[] h_FIs;
    delete[] h_FTs;
    delete[] h_fixLens;
}


ProbabilityData aDDM::computeGPUNLL(std::vector<aDDMTrial> trials, bool debug, int trialsPerThread, int timeStep, float approxStateStep) {
    int numTrials = trials.size();

    aDDMTrial* d_trials;
    double *d_likelihoods;
    cudaMalloc((void **) &d_trials, numTrials * sizeof(aDDMTrial));
    cudaMalloc((void **) &d_likelihoods, numTrials * sizeof(double));
    cudaMemcpy(d_trials, trials.data(), numTrials * sizeof(aDDMTrial), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256; 
    int numBlocks = 16; 

    aDDM::callGetTrialLikelihoodKernel(
        debug, trialsPerThread, numBlocks, threadsPerBlock,
        trials.data(), d_likelihoods, numTrials, 
        d, sigma, theta, barrier, 
        nonDecisionTime, timeStep, approxStateStep, decay
    );

    std::vector<double> h_likelihoods(numTrials);
    cudaMemcpy(h_likelihoods.data(), d_likelihoods, numTrials * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_trials);
    cudaFree(d_likelihoods);

    double NLL = 0;
    double likelihood = 0; 
    for (int i = 0; i < numTrials; i++) {
        likelihood += h_likelihoods[i];
        NLL += -log(h_likelihoods[i]);
    }
    ProbabilityData data = ProbabilityData(likelihood, NLL);
    data.trialLikelihoods = h_likelihoods; 
    return data; 
}
