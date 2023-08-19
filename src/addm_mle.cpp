#include <iostream>
#include <vector> 
#include <random>
#include <fstream>
#include <ctime>
#include <addm/gpu_toolbox.h>


using namespace std::chrono;

std::vector<float> rangeD = {0.0035, 0.005, 0.0065, 0.008};
std::vector<float> rangeSigma = {0.06, 0.065, 0.07, 0.075};
std::vector<float> rangeTheta = {0.35, 0.5, 0.65, 0.8};

int barrier = 1;

int main() {
    std::vector<aDDMTrial> trials = aDDMTrial::loadTrialsFromCSV("results/addm_simulations.csv");
    auto start = high_resolution_clock::now(); 
    MLEinfo info = aDDM::fitModelMLE(trials, rangeD, rangeSigma, rangeTheta, "thread", true);
    auto stop = high_resolution_clock::now(); 
    auto duration = duration_cast<seconds>(stop - start);
    std::cout << 
    "  Optimal Parameters  \n" << 
    "======================\n" <<
    "d      : " << info.optimal.d << "\n" << 
    "sigma  : " << info.optimal.sigma << "\n" << 
    "theta  : " << info.optimal.theta << "\n" << 
    "time   : " << duration.count() << std::endl;

    std::cout << info.likelihoods.size() << std::endl; 

    std::ofstream fp; 
    fp.open("results/addm_posteriors.csv"); 
    fp << "d,sigma,theta,p" << std::endl; 
    for (auto &i : info.likelihoods) {
        fp << i.first.d << "," << i.first.sigma << "," << i.first.theta << "," << i.second << std::endl; 
    }
}