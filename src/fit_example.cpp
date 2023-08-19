#include <addm/gpu_toolbox.h>
#include <vector>

int main() {
    std::map<int, std::vector<aDDMTrial>> data = loadDataFromCSV(
        "addm_code_and_data/Experiment1/python/expdata.csv", 
        "addm_code_and_data/Experiment1/python/fixations.csv"
    );
    std::vector<aDDMTrial> trials; 
    for (auto &i : data) {
        for (int j = 0; j < i.second.size(); j++) {
            if (j % 2 == 0) {
                trials.push_back(i.second[j]);
            }
        }
    }
    std::cout << "trials " << trials.size() << std::endl; 
    MLEinfo info = aDDM::fitModelMLE(trials, {0.003, 0.006, 0.009}, {0.03, 0.06, 0.09}, {0.3, 0.5, 0.7}, "thread");
    std::cout << "optimal " << info.optimal.d << " " << info.optimal.sigma << " " << info.optimal.theta << std::endl; 
}