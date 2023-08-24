#include <addm/gpu_toolbox.h>
#include <iostream>

using namespace std::chrono; 

int main() {
    std::vector<aDDMTrial> trials = aDDMTrial::loadTrialsFromCSV("results/addm_simulations.csv");

    auto start = high_resolution_clock::now(); 
    aDDM::fitModelMLE(trials, {0.003, 0.005}, {0.03, 0.07}, {0.5, 0.9}, "thread");
    auto stop = high_resolution_clock::now(); 
    auto duration_2 = duration_cast<milliseconds>(stop - start);

    start = high_resolution_clock::now(); 
    aDDM::fitModelMLE(trials, {0.003, 0.005, 0.007}, {0.03, 0.07, 0.09}, {0.3, 0.5, 0.9}, "thread");
    stop = high_resolution_clock::now(); 
    auto duration_3 = duration_cast<milliseconds>(stop - start);

    start = high_resolution_clock::now(); 
    aDDM::fitModelMLE(trials, {0.003, 0.005, 0.007, 0.009}, {0.03, 0.05, 0.07, 0.09}, {0.1, 0.3, 0.5, 0.9}, "thread");
    stop = high_resolution_clock::now(); 
    auto duration_4 = duration_cast<milliseconds>(stop - start);

    start = high_resolution_clock::now(); 
    aDDM::fitModelMLE(trials, {0.002, 0.003, 0.005, 0.007, 0.009}, {0.01, 0.03, 0.05, 0.07, 0.09}, {0, 0.1, 0.3, 0.5, 0.9}, "thread");
    stop = high_resolution_clock::now(); 
    auto duration_5 = duration_cast<milliseconds>(stop - start);

    start = high_resolution_clock::now(); 
    aDDM::fitModelMLE(trials, {0.001, 0.002, 0.003, 0.005, 0.007, 0.009}, {0.01, 0.03, 0.05, 0.07, 0.09, 0.11}, {0, 0.1, 0.3, 0.5, 0.9, 1}, "thread");
    stop = high_resolution_clock::now(); 
    auto duration_6 = duration_cast<milliseconds>(stop - start);

    start = high_resolution_clock::now(); 
    aDDM::fitModelMLE(trials, {0.001, 0.002, 0.003, 0.005, 0.006, 0.007, 0.009}, {0.01, 0.03, 0.05, 0.07, 0.09, 0.11, 0.13}, {0, 0.1, 0.3, 0.5, 0.6, 0.9, 1}, "thread");
    stop = high_resolution_clock::now(); 
    auto duration_7 = duration_cast<milliseconds>(stop - start);

    start = high_resolution_clock::now(); 
    aDDM::fitModelMLE(trials, {0.001, 0.002, 0.003, 0.005, 0.006, 0.007, 0.009, 0.011}, {0.01, 0.03, 0.05, 0.07, 0.09, 0.11, 0.13, 0.15}, {0, 0.1, 0.3, 0.5, 0.6, 0.8, 0.9, 1}, "thread");
    stop = high_resolution_clock::now(); 
    auto duration_8 = duration_cast<milliseconds>(stop - start);

    start = high_resolution_clock::now(); 
    aDDM::fitModelMLE(trials, {0.001, 0.002, 0.003, 0.005, 0.006, 0.007, 0.009, 0.011, 0.013}, {0.01, 0.03, 0.05, 0.07, 0.09, 0.11, 0.13, 0.15, 0.17}, {0, 0.1, 0.3, 0.4, 0.5, 0.6, 0.8, 0.9, 1}, "thread");
    stop = high_resolution_clock::now(); 
    auto duration_9 = duration_cast<milliseconds>(stop - start);

    std::cout << "=====TIMES=====" << std::endl <<
                 "2: " << duration_2.count() << std::endl <<
                 "3: " << duration_3.count() << std::endl <<
                 "4: " << duration_4.count() << std::endl <<
                 "5: " << duration_5.count() << std::endl << 
                 "6: " << duration_6.count() << std::endl << 
                 "7: " << duration_7.count() << std::endl << 
                 "8: " << duration_8.count() << std::endl <<
                 "9: " << duration_9.count() << std::endl;
}