#include <iostream>
#include <vector> 
#include <random>
#include <ctime>
#include <fstream>
#include <chrono> 
#include <sstream>
#include <addm/gpu_toolbox.h>


using namespace std::chrono;

int N = 1000;
float barrier = 1.0;
std::ofstream fp; 

std::vector<int> valDiffs = {-3, -2, -1, 0, 1, 2, 3};

std::vector<aDDMTrial> simulateDataset(float d, float sigma, float theta, float barrier, FixationData fixationData) {
    std::vector<aDDMTrial> outputs; 
    aDDM addm = aDDM(d, sigma, theta, barrier);

    for (int i = 0; i < N; i++) {
        std::mt19937 generator(std::random_device{}());
        std::uniform_int_distribution<std::size_t> distribution(0, valDiffs.size() - 1);
        int rIDX = distribution(generator);

        int valDiff = valDiffs.at(rIDX);
        int valueLeft = 3;
        int valueRight = valueLeft - valDiff;
        aDDMTrial adt = addm.simulateTrial(valueLeft, valueRight, fixationData);
        outputs.push_back(adt);
    }   
    return outputs; 
}

void testMLEfullGrid(FixationData fixationData) {
    std::vector<float> rangeD = {0.005, 0.006, 0.007};
    std::vector<float> rangeSigma = {0.05, 0.06, 0.07};
    std::vector<float> rangeTheta = {0.5, 0.6, 0.7};

    std::vector<aDDM> testModels;
    for (float d : rangeD) {
        for (float sigma : rangeSigma) {
            for (float theta : rangeTheta) {
                testModels.push_back(aDDM(d, sigma, theta, barrier));
            }
        }
    }

    for (float d : rangeD) {
        for (float sigma : rangeSigma) {
            for (float theta : rangeTheta) {
                std::cout << "Testing d=" << d << " sigma=" << sigma << " theta=" << theta << std::endl;
                std::vector<aDDMTrial> dataset = simulateDataset(d, sigma, theta, barrier, fixationData);

                double minNLL = __DBL_MAX__;
                float minD = 0; 
                float minSigma = 0; 
                float minTheta = 0;

                auto start = high_resolution_clock::now();
                for (aDDM addm : testModels) {
                    double NLL = addm.computeParallelNLL(dataset).NLL;
                    if (NLL < minNLL) {
                        minNLL = NLL;
                        minD = addm.d;
                        minSigma = addm.sigma;
                        minTheta = addm.theta;
                    }
                }
                auto stop = high_resolution_clock::now();
                auto duration = duration_cast<seconds>(stop - start);
                std::cout << "Time: " << duration.count() << std::endl;

                fp << "d=" << d << " sigma=" << sigma << " theta=" << theta << ",";
                if (minD == d && minSigma == sigma && minTheta == theta) {
                    fp << 1 << ",";
                    std::cout << "Test passed" << std::endl;
                } else {
                    fp << 0 << ",";
                    std::cout << "Test failed" << std::endl;
                    std::cout << "    Selected: d=" << minD << " sigma=" << minSigma << " theta=" << minTheta << std::endl;
                }
                fp << duration.count() << std::endl;
            }
        }
    }
}

void testDDMtimeSpeedup() {
    float d = 0.005;
    float sigma = 0.07;
    DDM ddm = DDM(d, sigma, barrier);
    std::vector<DDMTrial> trials; 

    for (int i = 0; i < N; i++) {
        std::mt19937 generator(std::random_device{}());
        std::uniform_int_distribution<std::size_t> distribution(0, valDiffs.size() - 1);
        int rIDX = distribution(generator);

        int valDiff = valDiffs.at(rIDX);
        int valueLeft = 3;
        int valueRight = valueLeft - valDiff;
        trials.push_back(ddm.simulateTrial(valueLeft, valueRight));
    }

    std::cout << "Testing basic implementation" << std::endl; 
    auto start = high_resolution_clock::now(); 
    double NLL = 0; 
    for (int i = 0; i < N; i++) {
        NLL += -log(ddm.getTrialLikelihood(trials[i]));
    }
    auto stop = high_resolution_clock::now(); 
    auto duration = duration_cast<milliseconds>(stop - start);
    std::cout << "Time: " << duration.count() << std::endl;  
    fp << "basic," << NLL << "," << duration.count() << std::endl; 

    std::cout << "Testing CPU Multithreaded implementation" << std::endl; 
    start = high_resolution_clock::now();
    NLL = ddm.computeParallelNLL(trials).NLL;
    stop = high_resolution_clock::now();
    duration = duration_cast<milliseconds>(stop - start);
    std::cout << "Time: " << duration.count() << std::endl;  
    fp << "multithread," << NLL << "," << duration.count() << std::endl;       

    std::cout << "Testing GPU implementation" << std::endl; 
    start = high_resolution_clock::now();
    NLL = ddm.computeGPUNLL(trials).NLL;
    stop = high_resolution_clock::now();
    duration = duration_cast<milliseconds>(stop - start);
    std::cout << "Time: " << duration.count() << std::endl;  
    fp << "GPU," << NLL << "," << duration.count() << std::endl;
}


void testSingleCSVLoad(FixationData fixationData) {
    float d = 0.005; 
    float sigma = 0.07; 
    float theta = 0.5; 
    int N = 1000; 

    aDDM addm = aDDM(d, sigma, theta, barrier);
    std::vector<aDDMTrial> trials = simulateDataset(d, sigma, theta, barrier, fixationData);
    aDDMTrial::writeTrialsToCSV(trials, "results/test_trials.csv");
    double likelihood = addm.computeParallelNLL(trials).likelihood;
    std::cout << "Likelihood: " << likelihood << std::endl; 

    std::map<int, std::vector<aDDMTrial>> data = loadDataFromSingleCSV("results/test_trials.csv");
    std::vector<aDDMTrial> newTrials = data.at(0);
    double newLikelihood = addm.computeParallelNLL(newTrials).likelihood;
    std::cout << "New Likelihood: " << newLikelihood << std::endl; 
    if (newLikelihood == likelihood) {
        std::cout << "Data load test passed" << std::endl; 
    } else {
        std::cout << "Data load test failed" << std::endl; 
    }
}

int main() {
    srand(time(NULL));
    
    fp.open("results/test_addm.csv");
    fp << "test,correctness,time\n";

    std::cout << "Testing MLE Full Grid" << std::endl;
    std::map<int, std::vector<aDDMTrial>> data = loadDataFromCSV("data/expdata.csv", "data/fixations.csv");
    FixationData fixationData = getEmpiricalDistributions(data);
    testMLEfullGrid(fixationData);
    fp.close();

    std::cout << "Testing time improvemnt for DDM" << std::endl; 
    fp.open("results/test_ddm.csv");
    fp << "test,NLL,time\n";
    testDDMtimeSpeedup();

    std::cout << "Testing single-CSV input for aDDM" << std::endl; 
    testSingleCSVLoad(fixationData);
}