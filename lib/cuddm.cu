#include <cuda.h>
#include <cuda_runtime.h>
#include <cassert>
#include <fstream>
#include "ddm.h"
#include "util.h"
#include "cuda_util.cuh"


__global__
void getTrialLikelihoodKernel(
    bool debug, 
    int trialsPerThread, 
    int *RTs, 
    int *choices, 
    int *valDiffs, 
    double* likelihoods,
    int numTrials, 
    float *states, 
    int biasState,
    int numStates, 
    float stateStep, 
    float d, 
    float sigma, 
    int barrier, 
    int nonDecisionTime, 
    int timeStep, 
    float approxStateStep, 
    float dec) {

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < numTrials / trialsPerThread) {
        double *prStates = new double[numStates];
        float *changeMatrix = new float[numStates * numStates];
        float *probDistChangeMatrix = new float[numStates * numStates];
        double* prStatesNew = new double[numStates];
        float *changeUpCDFs = new float[numStates];
        float *changeDownCDFs = new float[numStates];

        for (int trialNum = tid * trialsPerThread; trialNum < (tid + 1) * trialsPerThread; trialNum++) {
            int choice = choices[trialNum];
            int RT = RTs[trialNum];
            int valDiff = valDiffs[trialNum];


            int numTimeSteps = RT / timeStep; 

            // requires compute capability 2.x
            float* barrierUp = new float[numTimeSteps];
            float *barrierDown = new float[numTimeSteps];

            for (int i = 0 ; i < numTimeSteps; i++) {
                barrierUp[i] = barrier / (1 + (dec * i));
                barrierDown[i] = -barrier / (1 + (dec * i));
            }

            for (int i = 0; i < numStates; i++) {
                prStates[i] = (i == biasState) ? 1 : 0; 
            }

            double *probUpCrossing = new double[numTimeSteps];
            double *probDownCrossing = new double[numTimeSteps];
            for (int i = 0; i < numTimeSteps; i++) {
                probUpCrossing[i] = 0; 
                probDownCrossing[i] = 0; 
            }

            if (debug) {
                for (int i = 0 ; i < numStates ; i++) {
                    printf("prStates[%i] = %f\n", i, prStates[i]);
                }
            }
            
            for (int i = 0; i < numStates; i++) {
                for (int j = 0; j < numStates; j++) {
                    changeMatrix[__RC2IDX(i, j, numStates)] = states[i] - states[j];
                }
            }

            float *changeUp = new float[numStates * numTimeSteps];
            for (int i = 0; i < numStates; i++) {
                for (int j = 0; j < numTimeSteps; j++) {
                    changeUp[__RC2IDX(i, j, numTimeSteps)] = barrierUp[j] - states[i];
                }
            }

            float *changeDown = new float[numStates * numTimeSteps];
            for (int i = 0; i < numStates; i++) {
                for (int j = 0; j < numTimeSteps; j++) {
                    changeDown[__RC2IDX(i, j, numTimeSteps)] = barrierDown[j] - states[i];
                }
            }

            if (debug) {
                printf("change matrix\n");
                for (int i = 0; i < numStates * numStates; i++) {
                    printf("%f ", changeMatrix[i]);
                    if ((i + 1) % numStates == 0) {
                        printf("\n");
                    }
                }

                printf("change up\n");
                for (int i = 0; i < numStates * numTimeSteps; i++) {
                    printf("%f ", changeUp[i]);
                    if ((i + 1) % numTimeSteps == 0) {
                        printf("\n");
                    }
                }
            }


            int elapsedNDT = 0;
            bool recomputePDCM = true; 
            float prevMean = 0; 
            for (int time = 1; time < numTimeSteps; time++) {

                if (debug) printf(
                    "============\n timestep %i \n============", time
                );

                float mean; 
                if (elapsedNDT < nonDecisionTime / timeStep) {
                    mean = 0; 
                    elapsedNDT += 1; 
                } else {
                    mean = d * valDiff;
                }

                if (mean != prevMean) {
                    recomputePDCM = true;
                } else {
                    recomputePDCM = false; 
                }

                if (recomputePDCM || time == 1) {
                    for (int i = 0; i < numStates; i++) {
                        for (int j = 0; j < numStates; j++) {
                            float x = changeMatrix[__RC2IDX(i, j, numStates)];
                            probDistChangeMatrix[__RC2IDX(i, j, numStates)] = __pdf(x, mean, sigma);
                        }
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

                double rowSum; 
                for (int i = 0; i < numStates; i++) {
                    rowSum = 0; 
                    for (int j = 0; j < numStates; j++) {
                        rowSum += stateStep * probDistChangeMatrix[__RC2IDX(i, j, numStates)] * prStates[j];
                    }
                    prStatesNew[i] = (states[i] > barrierUp[time] || states[i] < barrierDown[time]) ? 0 : rowSum;
                }

                if (debug) {
                    for (int i = 0 ; i < numStates ; i++) {
                        printf("prStatesNew[%i] = %f\n", i, prStatesNew[i]);
                    }
                }

                for (int i = 0; i < numStates; i++) {
                    float x = changeUp[__RC2IDX(i, time, numTimeSteps)];
                    changeUpCDFs[i] = 1 - normcdff((x - mean) / sigma);
                }
                if (debug) {
                    for (int i = 0; i < numStates; i++) {
                        printf("changeUpCDFs[%i] = %f\n", i, changeUpCDFs[i]);
                    }
                }
                double tempUpCross = 0; 
                for (int i = 0; i < numStates; i++) {
                    tempUpCross += changeUpCDFs[i] * prStates[i];
                }

                for (int i = 0; i < numStates; i++) {
                    float x = changeDown[__RC2IDX(i, time, numTimeSteps)];
                    changeDownCDFs[i] = normcdff((x - mean) / sigma);
                }
                if (debug) {
                    for (int i = 0; i < numStates; i++) {
                        printf("changeDownCDFs[%i] = %f\n", i, changeDownCDFs[i]);
                    }
                }
                double tempDownCross = 0; 
                for (int i = 0; i < numStates; i++) {
                    tempDownCross += changeDownCDFs[i] * prStates[i];
                }

                if (debug) printf("temp up cross = %f\n", tempUpCross);
                if (debug) printf("temp down cross = %f\n", tempDownCross);

                double sumIn = 0; 
                double sumCurrent = tempUpCross + tempDownCross; 
                for (int i = 0; i < numStates; i++) {
                    sumIn += prStates[i];
                    sumCurrent += prStatesNew[i];
                }
                double normFactor = sumIn / sumCurrent; 
                for (int i = 0; i < numStates; i++) {
                    prStates[i] = prStatesNew[i] * normFactor; 
                }

                probUpCrossing[time] = tempUpCross * normFactor; 
                probDownCrossing[time] = tempDownCross * normFactor;

                prevMean = mean;
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

        delete[] prStates;
        delete[] changeMatrix;            
        delete[] probDistChangeMatrix;
        delete[] prStatesNew;
        delete[] changeUpCDFs;
        delete[] changeDownCDFs;
    }    
}

void DDM::callGetTrialLikelihoodKernel(
    bool debug, int trialsPerThread, int numBlocks, int threadsPerBlock, 
    DDMTrial *trials, double *likelihoods, 
    int numTrials, float d, float sigma, float barrier, 
    int nonDecisionTime, int timeStep, float approxStateStep, float dec) {

    int *d_RTs, *d_choices, *d_VDs;
    cudaMalloc((void**)&d_RTs, numTrials * sizeof(int));
    cudaMalloc((void**)&d_choices, numTrials * sizeof(int));
    cudaMalloc((void**)&d_VDs, numTrials * sizeof(int));

    int *h_VDs = new int[numTrials];
    int *h_RTs = new int[numTrials];
    int *h_choices = new int[numTrials];
    for (int i = 0; i < numTrials; i++) {
        h_VDs[i] = trials[i].valueLeft - trials[i].valueRight;
        h_RTs[i] = trials[i].RT;
        h_choices[i] = trials[i].choice;
    }

    cudaMemcpy(d_RTs, h_RTs, numTrials * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_choices, h_choices, numTrials * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_VDs, h_VDs, numTrials * sizeof(int), cudaMemcpyHostToDevice);
    
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

    getTrialLikelihoodKernel<<<numBlocks, threadsPerBlock>>>(
        debug,
        trialsPerThread,
        d_RTs,
        d_choices,
        d_VDs,
        likelihoods,
        numTrials,
        d_states, 
        biasState,
        numStates,
        stateStep,
        d, sigma, barrier,
        nonDecisionTime,
        timeStep,
        approxStateStep,
        dec
    );

    cudaFree(d_RTs);
    cudaFree(d_choices);
    cudaFree(d_VDs);
    cudaFree(d_states);
    delete[] h_RTs;
    delete[] h_choices;
    delete[] h_VDs;
    delete[] states;
    }
        

ProbabilityData DDM::computeGPUNLL(std::vector<DDMTrial> trials, bool debug, int trialsPerThread, int timeStep, float approxStateStep) {
    int numTrials = trials.size(); 

    DDMTrial *d_trials;
    double *d_likelihoods;
    cudaMalloc((void**) &d_trials, numTrials * sizeof(DDMTrial));
    cudaMalloc((void**) &d_likelihoods, numTrials * sizeof(double));
    cudaMemcpy(d_trials, trials.data(), numTrials * sizeof(DDMTrial), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256; 
    int numBlocks = 16;

    DDM::callGetTrialLikelihoodKernel(
        debug, trialsPerThread, numBlocks, threadsPerBlock, 
        trials.data(), d_likelihoods, 
        numTrials, d, sigma, barrier, 
        nonDecisionTime, timeStep, approxStateStep, decay);

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
