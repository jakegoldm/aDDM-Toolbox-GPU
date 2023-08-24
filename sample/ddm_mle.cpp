#include <iostream>
#include <vector> 
#include <fstream> 
#include <addm/gpu_toolbox.h>

// Location of the DDM simulations. 
const std::string SIMS = "results/ddm_simulations.csv"; 
// Location to save the computed likelihoods to. 
const std::string SAVE = "results/ddm_likelihoods.csv"; 
// Parameter ranges. Change as desired. 
const std::vector<float> rangeD = {0.004, 0.005, 0.006};
const std::vector<float> rangeSigma = {0.05, 0.07, 0.09};

int main() {
    // Load trials from a CSV. 
    std::vector<DDMTrial> trials = DDMTrial::loadTrialsFromCSV(SIMS); 
    // Add additional arguments to specify computation mode, etc.. if desired. 
    MLEinfo<DDM> info = DDM::fitModelMLE(trials, rangeD, rangeSigma);
    std::cout << 
    "  Optimal Parameters  \n" << 
    "======================\n" <<
    "d      : " << info.optimal.d << "\n" << 
    "sigma  : " << info.optimal.sigma << std::endl; 

    // Save computed likelihoods to CSV. 
    std::ofstream fp;
    fp.open(SAVE);
    fp << "d,sigma,p\n";
    for (auto i : info.likelihoods) {
        fp << i.first.d << "," << i.first.sigma << "," << i.second << "\n";
    }
    fp.close();

}