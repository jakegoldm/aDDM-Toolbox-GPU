# aDDM-Toolbox-GPU #

This toolbox can be used to perform model fitting and data simulation for the Drift Diffusion Model (DDM) and the attentional Drift Diffusion Model (aDDM). It is aimed to provide computational speedup, employing GPU optimizations for parameter estimation. 

<!-- Update TOC with `doctoc .`-->
## Table of Contents ## 

<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->

- [Requirements](#requirements)
- [Installation and Usage](#installation-and-usage)
- [Examples](#examples)
  - [N-Trial Simulation](#n-trial-simulation)
  - [Trial Likelihood Compuation](#trial-likelihood-compuation)
  - [Negative Log Likelihoods (NLL) and Maximum Likelihood Estimation (MLE)](#negative-log-likelihoods-nll-and-maximum-likelihood-estimation-mle)
- [Python Bindings](#python-bindings)
  - [Optional: Python Syntax Highlighting](#optional-python-syntax-highlighting)
- [Sample C++ Scripts](#sample-c-scripts)
- [Data Analysis Scripts](#data-analysis-scripts)
- [Authors](#authors)
- [Acknowledgements](#acknowledgements)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

## Requirements ##

This library requires __NVCC__ and CUDA Toolkit versions 12.2 and __g++__ version 11.3.0. This library also uses three third-party C++ packages for thread pools, JSON processing, and statistical distributions: 

* [BS::thread_pool](https://github.com/bshoshany/thread-pool)
* [JSON for Modern C++](https://github.com/nlohmann/json)
* [Boost Math/Statistical Distributions](https://www.boost.org/doc/libs/?view=category_math)

These dependencies can be installed using the following commands: 

```shell
$ wget -O /usr/include/c++/11/BS_thread_pool.hpp https://raw.githubusercontent.com/bshoshany/thread-pool/master/include/BS_thread_pool.hpp
$ mkdir -p /usr/include/c++/11/nlohmann
$ wget -O /usr/include/c++/11/nlohmann/json.hpp https://raw.githubusercontent.com/nlohmann/json/develop/single_include/nlohmann/json.hpp
$ apt-get install libboost-math-dev libboost-math1.74-dev
```

*Note that the installation directory /usr/include/c++/11 may be modified to support newer versions of C++. In the event of a __Permission Denied__ error, precede the above commands with __sudo__.*

For Python data visualization, the base requirement is Python 3.10. The following libraries are required: 

* matplotlib 
* numpy 
* pandas
* seaborn 

See [requirements.txt](requirements.txt) for specific versions. These libraries can be installed using the following command:

```shell
$ pip install matplotlib numpy pandas seaborn
```

## Installation and Usage ## 

The aDDM-Toolbox-GPU library offers both a GPU and non-GPU reliant installation target. The default installation build employs CUDA libraries, including the NVCC compiler. For systems without CUDA capabilities or users looking to exclude CUDA code from the installation, uncomment the following line in the Makefile: 

```
MACROS := -DEXCLUDE_CUDA_CODE
```

This macro declaration will ensure that all instances of CUDA code are undefined in the installation. Note that this removes the necessity to have NVCC and CUDA versions 12.2 installed; however, all other dependencies are still required. The aDDM-Toolbox-GPU library can then be built and installed in one step: 

```shell
$ make install
```

*In the event of a __Permission Denied__ error, precede the above command with __sudo__.*

This will install the libaddm.so shared library as well as the corresponding header files. Although there are multiple header files corresponding to the aDDM and DDM programs, simply adding `#include <addm/gpu_toolbox.h>` to a C++/CUDA program will include all necessary headers. A simple usage example is described below: 

`main.cpp`:
```C++
#include <addm/gpu_toolbox.h>
#include <iostream>

int main() {
    aDDM addm = aDDM(0.005, 0.07, 0.5, 1);
    std::cout << "d: " << addm.d << std::endl; 
    std::cout << "sigma: " << addm.sigma << std::endl; 
    std::cout << "theta: " << addm.theta << std::endl; 
}
```

When compiling any code using the toolbox, include the `-laddm` flag to link with the installed shared object library.

```
$ nvcc -o main main.cpp -laddm
$ ./main
d: 0.005
sigma: 0.07
theta: 0.5
```

*Note that __nvcc__ is assumed to be at least __version 12.2__. The compiler path may need to be modified to support this requirement. For this example, or if using the CPU-dependent build, __nvcc__ may be substituted with __g++__.*

## Examples ##

For any data simulation and model fitting, the DDM and aDDM classes are going to be the most useful. The `addm.h` amd `ddm.h` files can be found in the include directory. Below are some examples that may give you an idea how to use these classes. Besides these examples, you may want to:

* Check out the [documentation](https://jakegoldm.github.io/aDDM-Toolbox-GPU/).
* See example use cases in the [sample directory](sample/).

### N-Trial Simulation ###

Both the `DDM` and `aDDM` classes provide methods for simulating mass amounts of trials. This can be done using the `simulateTrial` method defined in both classes: 

__DDM__
```C++
#include <addm/gpu_toolbox.h>
#include <vector>

using namespace std; 

int N = 1000; 
int valueLeft = 3; 
int valueRight = 7; 
int SEED = 540; 

int main() {
    vector<DDMTrial> trials;
    DDM ddm = DDM(0.005, 0.07, 1);

    for (int i = 0; i < N; i++) {
        DDMTrial trial = ddm.simulateTrial(valueLeft, valueRight, 10, SEED);
        trials.push_back(trial);
    }

    DDMTrial::writeTrialsToCSV(trials, "ddm_example.csv");
}
```
`ddm_example.csv`:
```
choice,rt,valueLeft,valueRight
1,850,3,7
...
```

__aDDM__

Note that for generating aDDM trials, an existing set of empirical fixations is required. This data can be packaged in the form of 1-2 CSVs. The input format for a single CSV is identical to the output of `aDDMTrial::writeTrialsToCSV(...)`. This CSV should be formatted as follows: 

|  trial 	|choice |   rt	|  valueLeft 	|  valueRight 	|  fixItem 	|  fixTime 	|
|:-:	|:-:	|:-:	|:-:	        |:-:	        |:-:	    |:-:	    |
|   0	|  1 	|  350 	|   3	        |   0           |   0	    |   200	    |
|   0	|   1	|  350 	|   3	        |   0	        |   1	    |   150	    |
|   1	|   -1	|  400 	|   4	        |  5            |   0	    |   300	    |
|   1	|   -1	|  400 	|   4	        |   5	        |   2	    |   100	    |

To load data using the single-CSV format: 

```C++
#include <addm/gpu_toolbox.h>
#include <vector>

using namespace std; 

int N = 1000; 
int valueLeft = 3; 
int valueRight = 7; 
int SEED = 540; 

int main() {
    map<int, vector<aDDMTrial>> data = loadDataFromSingleCSV("data/addm_sims.csv");
    FixationData fixationData = getEmpiricalDistributions(data);

    vector<aDDMTrial> trials;
    aDDM addm = aDDM(0.005, 0.07, 0.5, 1);

    for (int i = 0; i < N; i++) {
        aDDMTrial trial = addm.simulateTrial(
            valueLeft, valueRight, fixationData, 10, 3, {}, {}, SEED);
        trials.push_back(trial);
    }

    aDDMTrial::writeTrialsToCSV(trials, "addm_example.csv");
}
```
`addm_example.csv`:
```
trial,choice,rt,valueLeft,valueRight,fixItem,fixTime
0,1,1850,3,7,0,280
0,1,1850,3,7,1,230
0,1,1850,3,7,0,70
...
```

In the case of using two CSVs, such as with real experimental data, the first CSV should contain the experimental data including subject parcode, trial ID, RT, choice, and item values. The second CSV should contain fixation data pertaining to each trial. Sample CSV files in an acceptable format can be found in the data directory. 

__Experimental Data__

| parcode | trial | rt | choice | valueLeft | valueRight |  
| :-:     | :-:   |:-: | :-:    | :-:       | :-:        | 
| 0       | 0     |1962| -1     | 15        | 0          | 
| 0       | 1     |873 | 1      | -15       | 5          |  
| 0       | 2     |1345| 1      | 10        | -5         |  

__Fixation Data__

| parcode | trial | fixItem | fixTime |
| :-:     | :-:   | :-:      | :-:      |
| 0       | 0     | 3        | 176      | 
| 0       | 0     | 0        | 42       | 
| 0       | 0     | 1        | 188      | 

To load data using the two-CSV format: 

```C++
#include <addm/gpu_toolbox.h>
#include <vector>

using namespace std; 

int N = 1000; 
int valueLeft = 3; 
int valueRight = 7; 
int SEED = 540; 

int main() {
    map<int, vector<aDDMTrial>> data = loadDataFromCSV("data/expdata.csv", "data/fixations.csv");
    FixationData fixationData = getEmpiricalDistributions(data);

    vector<aDDMTrial> trials;
    aDDM addm = aDDM(0.005, 0.07, 0.5, 1);

    for (int i = 0; i < N; i++) {
        aDDMTrial trial = addm.simulateTrial(
            valueLeft, valueRight, fixationData, 10, 3, {}, {}, SEED);
        trials.push_back(trial);
    }
    
    aDDMTrial::writeTrialsToCSV(trials, "addm_example.csv");
}
```
`addm_example.csv`:
```
trial,choice,rt,valueLeft,valueRight,fixItem,fixTime
0,1,1510,3,7,0,370
0,1,1510,3,7,2,280
0,1,1510,3,7,0,70
...
```

### Trial Likelihood Compuation ###

The `DDM` and `aDDM` classes both provide functionality for estimating trial likelihoods. The `getTrialLikelihood(...)` method takes in a single trial object and returns the likelihood that the provided model will generate that trial. Examples are provided below: 

__DDM__ 

```C++
#include <addm/gpu_toolbox.h>
#include <iostream>

using namespace std; 

int main() {
    DDM ddm = DDM(0.005, 0.07, 1);
    DDMTrial trial = DDMTrial(2400, -1, 4, 5); 
    double prob = ddm.getTrialLikelihood(trial);
    cout << "Probability = " << prob << endl; 
}
```
Output: 
```
Probability = 0.000166691
```

__aDDM__

```C++
#include <addm/gpu_toolbox.h>
#include <iostream>

using namespace std; 

int main() {
    aDDM addm = aDDM(0.005, 0.07, 0.5, 1);
    aDDMTrial trial = aDDMTrial(
        240, 1, 3, 3, {0, -1}, {100, 140}
    );
    double prob = addm.getTrialLikelihood(trial);
    cout << "Probability = " << prob << endl; 
}
```
Output: 
```
Probability = 0.000515395
```

### Negative Log Likelihoods (NLL) and Maximum Likelihood Estimation (MLE) ###

A useful application of the trial likelihood computation is determining the total __Negative Log Likelihood (NLL)__ of an aggregation of trials. Each instance of the `DDM` and `aDDM`classes can use the following methods to determine the sum of NLLs for the trials. 

* `computeParallelNLL(...)` to compute the sum of NLLs for a vector of trials using CPU multithreading. 
* `computeGPUNLL(...)` to compute tthe sum of NLLs for a vector of trials using GPU parallelism. 

The sum of NLLs can also be computed without any optimization features using a simple for-loop and the `getTrialLikelihood` method. 

These methods can be applied for model fitting and performing __Maximum Likelihood Estimations (MLE)__ for a set of models. The `fitModelMLE(...)` method of both the `DDM` and `aDDM` allows a user to select a range of available parameters to test for, as well as the computational optimizations to employ based off the user's hardware. The `fitModelMLE(...)` method returns an instance of the `MLEinfo` struct, which contains the model with the most optimal parameters and aggregated likelihoods information: 

1. Mapping of models to their calculated NLLs over the dataset of trials. 
2. Mapping of models to their normalized posteriors, taken as the sum of likelihoods. 

Examples of MLE calculations are described below: 

__DDM__

```C++
#include <addm/gpu_toolbox.h>
#include <iostream>

using namespace std;

vector<float> rangeD = {0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009};
vector<float> rangeSigma = {0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09};

int barrier = 1;

int main() {
    vector<DDMTrial> trials = DDMTrial::loadTrialsFromCSV("data/ddm_sims.csv"); 
    cout << "Counted " << trials.size() << " trials." << endl;
    MLEinfo<DDM> info = DDM::fitModelMLE(trials, rangeD, rangeSigma, barrier, "gpu", true);
    cout << "Optimal d=" << info.optimal.d << " sigma=" << info.optimal.sigma << endl; 
}
```
Output: 
```
Counted 1000 trials.
Optimal d=0.005 sigma=0.07
```

__aDDM__

```C++
#include <addm/gpu_toolbox.h>
#include <iostream>

using namespace std;

vector<float> rangeD = {0.003, 0.004, 0.005};
vector<float> rangeSigma = {0.06, 0.07, 0.08};
vector<float> rangeTheta = {0.5, 0.6, 0.7};

int barrier = 1;
int valueLeft = 3; 

int main() {
    vector<aDDMTrial> trials = aDDMTrial::loadTrialsFromCSV("data/addm_sims.csv");
    cout << "Counted " << trials.size() << " trials." << endl;
    MLEinfo info = aDDM::fitModelMLE(trials, rangeD, rangeSigma, rangeTheta, barrier, "thread", true);
    cout << "Optimal d=" << info.optimal.d << 
                 " sigma=" << info.optimal.sigma << 
                 " theta=" << info.optimal.theta << 
                 endl;    
}
```
Output:
```
Counted 1000 trials. 
Optimal d=0.005 sigma=0.07 theta=0.5
```

Both the `DDM` and  `aDDM` instances of `fitModelMLE` allow for the `bias` and `decay` fields of the model to be fit as well. See the documentation for instructions on how to do so. 


## Python Bindings ## 

Python bindings are also provided for users who prefer to work with a Python codebase over C++. The provided bindings are located in [lib/bindings.cpp](lib/bindings.cpp). Note that [pybind11](https://github.com/pybind/pybind11) and Python version 3.10 (at a minimum) are __strict__ prerequisites for installation and usage of the Python code. These can be installed with 

```shell
apt-get install python3.10
pip3 install pybind11
```

Once `pybind11` and Python3.10 are installed, the module can be built with:

```
make pybind
```

This will create a shared library object in the repository's root directory containing the `addm_toolbox_gpu` module. Although function calls remain largely analogous with the original C++ code, an example is described below that can be used to ensure the code is working properly: 

`main.py`: 

```Python
import addm_toolbox_gpu

ddm = addm_toolbox_gpu.DDM(0.005, 0.07)
print(f"d = {ddm.d}")
print(f"sigma = {ddm.sigma}")

trial = ddm.simulateTrial(3, 7, 10, 540)
print(f"RT = {trial.RT}")
print(f"choice = {trial.choice}")
```
To run the code: 
```
$ python3 main.py
d = 0.005
sigma = 0.07
RT = 850
choice = 1
```

### Optional: Python Syntax Highlighting ###

For users working in a user interface, such as Visual Studio Code, a Python stub is provided to facilitate features including syntax highlighting, type-hinting, auto-complete. Although the `addm_toolbox_gpu.pyi` stub is built-in, the file can be dynamically generated using the [mypy stubgen](https://mypy.readthedocs.io/en/stable/stubgen.html) tool. The `mypy` module can be installed using: 

```shell
pip install mypy
```

For users who plan to modify the library for their own use and want the provided features, the stub file can be built as follows: 

```shell
stubgen -m addm_toolbox_gpu -o .
```
*Note that the `pybind11` shared library file should be built before running `stubgen`.*

## Sample C++ Scripts ##

This repository provides a set of built-in scripts for running trial simulations and model-fitting. These scripts are provided in the [sample directory](sample/) and are described below: 

* [`ddm_simulate.cpp`](sample/ddm_simulate.cpp): Simulate N trials using the ddm. Build using the `make sim` rule. Specify an output CSV file in the file constants. Usage: 

```shell
bin/ddm_simulate [numTrials] [d] [sigma]
```

* [`addm_simulate.cpp`](sample/addm_simulate.cpp): Simulate N trials using the addm. Build using the `make sim` rule. Specify an output CSV file in the file constants. Sample files for empirical fixation distributions are provided in the [data](data/) directory. However, these should be specified by the user as well in the file constants. Usage: 

```shell
bin/addm_simulate [numTrials] [d] [sigma] [theta]
```

* [`ddm_mle.cpp`](sample/ddm_mle.cpp): Fit a DDM model's parameters. Build using the `make mle` rule. CSV simulations, output location, and parameter ranges should be specified in the file constants. Usage: 

```shell
bin/ddm_mle
```

* [`addm_mle.cpp`](sample/addm_mle.cpp): Fit an aDDM model's paramters. Build using the `make mle` rule. CSV simulations, output location, and parameter ranges should be specified int he file constants. Usage: 

```shell 
bin/addm_mle
```

## Data Analysis Scripts ##

A set of data analysis and visualization tools are provided in the [analysis](analysis/) directory. Current provided scripts include: 

* DDM Heatmaps for MLE. 
* aDDM Heatmap for MLE. 
* Posterior Pair Plots.
* Time vs RDV for individual trial.
* Value Differences against Response Time Frequencies. 

See the individual file documentation for usage instructions. 

## Authors ## 

Jake Goldman - jgoldman@caltech.edu, [jakegoldm](https://github.com/jakegoldm)

## Acknowledgements ##

This toolbox was developed as part of a resarch project in the [Rangel Neuroeconomics Lab](http://www.rnl.caltech.edu/) at the California Institute of Technology. Special thanks to Antonio Rangel and Zeynep Enkavi for your help with this project. 