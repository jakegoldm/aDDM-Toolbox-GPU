#include <iostream>
#include <vector> 
#include <random>
#include <fstream>
#include <addm/gpu_toolbox.h>

// Location to save the DDM trials. 
const std::string SAVE = "results/ddm_simulations.csv"; 
// Sample value differences. Change if desired. 
const std::vector<int> valDiffs = {-3, -2, -1, 0, 1, 2, 3};

/**
 * Example Usage: 
 * 
 * bin/ddm_simulate 1000 0.005 0.07
 */
int main(int argc, char** argv) {
    int N; 
    float d; 
    float sigma; 

    // Check the input arguments are valid. 
    if (argc != 4) {
        std::cerr << "Provide 3 arguments." << std::endl;
        exit(1);
    }
    try {
        N = stoi(argv[1]);
    } catch (invalid_argument &e) {
        std::cerr << "Input N not convertable to int: " << argv[1] << std::endl; 
        exit(1);
    }
    try {
        d = stof(argv[2]);
    } catch (invalid_argument &e) {
        std::cerr << "Input d not convertable to float: " << argv[2] << std::endl; 
        exit(1);
    }
    try {
        sigma = stof(argv[3]);
    } catch (invalid_argument &e) {
        std::cerr << "Input sigma not convertable to float: " << argv[3] << std::endl;  
        exit(1);
    }

    std::cout << "Performing " << N << " trials." << std::endl; 
    std::cout << "d=" << d << std::endl; 
    std::cout << "sigma=" << sigma << std::endl; 

    std::vector<DDMTrial> trials;
    srand(time(NULL));

    // Create a DDM with the specified parameters. 
    DDM ddm = DDM(d, sigma);

    // Create N trials with a random value difference. 
    std::mt19937 generator(std::random_device{}());
    std::uniform_int_distribution<std::size_t> distribution(0, valDiffs.size() - 1);
    for (int i = 0; i < N; i++) {
        int rIDX = distribution(generator);

        int valDiff = valDiffs.at(rIDX);
        int valueLeft = 3;
        int valueRight = valueLeft - valDiff;
        DDMTrial dt = ddm.simulateTrial(valueLeft, valueRight);
        trials.push_back(dt);
    }

    // Write trials to a CSV. 
    DDMTrial::writeTrialsToCSV(trials, SAVE);
}