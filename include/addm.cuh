#ifndef ADDM_CUH
#define ADDM_CUH

#include <string>
#include <vector>
#include <map>
#include <tuple>
#include "ddm.cuh"
#include "mle_info.h"


using namespace std;


using fixDists = map<int, vector<float>>; /**< Maps fixation numbers to the measured durations 
for that fixation in the provided data. */


/**
 * @brief Record of empirical fixation data that is used for simulating new aDDMTrials. 
 * 
 */
class FixationData {
    private:
    public:
        float probFixLeftFirst; /**< Float between 0 and 1 specifying the empirical probability 
            that the left item will be fixated on first. */
        vector<int> latencies; /**< Vector of ints corresponding to the empirical distribution 
            of trial latencies (Delay before first fixation) in milliseconds. */
        vector<int> transitions; /**< Vector of ints corresponding to the empirical distribution 
            of transitions (delays between item fixations) in milliseconds. */
        fixDists fixations; /**< Mapping of fixation numbers (i.e. first, second, etc...) to the
            empirical distribution of fixation durations for each number. */


        /**
         * @brief Construct a new Fixation Data object.
         * 
         * @param probFixLeftFirst Probability of fixating left first. 
         * @param latencies Empirical distribution of trial latencies. 
         * @param transitions Empirical distribution of trial transitions.
         * @param fixations Mapping of fixation numbers to empirical duration distributions. 
         */
        FixationData(
            float probFixLeftFirst, vector<int> latencies, 
            vector<int> transitions, fixDists fixations
        );
    };


/**
 * @brief Implementation of a single aDDMTrial object. 
 * 
 * An aDDMTrial can either be generated via simulating data trials or loaded from a CSV. A single
 * trial represents an individual binary perceptual choice made by one subject. These trials can 
 * be aggregated together for model fitting and likelihood computations. 
 *  
 */
class aDDMTrial: public DDMTrial {
    private:
    public:
        vector<int> fixItem; /**< Vector of integers representing the items fixated on during the 
            trial in chronological order. 1 corresponds to left, 2 corresponds to right, and any
            other value is considered a fixation or blank fixation. */
        vector<int> fixTime; /**< Vector of integers corresponding to the duration of each 
            fixation. */
        vector<float> fixRDV; /**< Vector of floats corresponding to the RDV values at the end of
            each fixation. */
        float uninterruptedLastFixTime; /**< Integer corresponding to the duration (milliseconds) 
            that the last fixation in the trial would have if it had not been terminated when a 
            decision had been made. */


        /**
         * @brief Construct a new aDDM Trial object.
         * 
         * @param RT Response time in milliseconds. 
         * @param choice Either -1 (for left item) or +1 (for right item).
         * @param valueLeft Value of the left item. 
         * @param valueRight Value of the right item. 
         * @param fixItem Vector of integers representing the items fixated on during the trial in
         * chronological order. 1 corresponds to left, 2 corresponds to right, and any other value
         * is considered a transition or blank fixation. 
         * @param fixTime Vector of integers corresponding to the duration of each fixation. Must 
         * be equal in size to fixItem. 
         * @param fixRDV Vector of floats corresopnding to the RDV values at the end of each 
         * fixation. 
         * @param uninterruptedLastFixTime Integer corresponding to the duration (milliseconds) 
         * that the last fixation in the trial would have if it had not been terminated when a 
         * decision had been made. 
         */
        aDDMTrial(
            unsigned int RT, int choice, int valueLeft, int valueRight, 
            vector<int> fixItem={}, vector<int> fixTime={}, 
            vector<float> fixRDV={}, float uninterruptedLastFixTime=0);


        /**
         * @brief Construct an empty aDDMTrial object. 
         * 
         */
        aDDMTrial() {};


        /**
         * @brief Write a vector of aDDMTrials to a CSV file. 
         * 
         * @param trials Vector of trials to be saved. 
         * @param filename File to store the trials in. 
         */
        static void writeTrialsToCSV(vector<aDDMTrial> trials, string filename);


        /**
         * @brief Load a dataset of aDDMTrials into program memory. 
         * 
         * @param filename Location of the data trials. 
         * @return vector<aDDMTrial> containing the stored trials. 
         */
        static vector<aDDMTrial> loadTrialsFromCSV(string filename);
};


/**
 * @brief Implementation of the attentional Drift Diffusion Model (aDDM). 
 * 
 * This class contains an implementation of the attentional Drif Diffusion Model (aDDM) as described
 * by Krajbich et al. (2010). It builds upon the standard and well-known model of the DDM by 
 * considering the impact of visual ffixation patterns on perceptual binary choices. This class 
 * provides methods for data simulation and model fitting via Maximum Likelihood Estimation. The 
 * process of model fitting involves selecting some range of potential parameters (d, theta, sigma)
 * and iterating over the parameter space to deterimine which combination best fits the provided 
 * set of aDDM trials. For details on the simulation and model fitting process, see the individual 
 * methods described below.  
 */
class aDDM: public DDM {
    private:
#ifndef EXCLUDE_CUDA_CODE
        void callGetTrialLikelihoodKernel(
            bool debug, int trialsPerThread, int numBlocks, int threadsPerBlock, 
            aDDMTrial *trials, double *likelihoods, int numTrials, 
            float d, float sigma, float theta, float barrier, 
            int nonDecisionTime, int timeStep, float approxStateStep, float decay);
#endif 


    public: 
        float theta; /**< Float between 0 and 1, parameter of the model which 
            controls the attentional bias.*/

        bool operator <( const aDDM &rhs ) const { return (d + sigma + theta < rhs.d + rhs.sigma + rhs.theta); }
        

        /**
         * @brief Construct a new aDDM object.
         * 
         * @param d Drift rate.
         * @param sigma Noise or standard deviation for the normal distribution.
         * @param theta Ranges on [0,1] and indicates level of attentional bias.
         * @param barrier Positive magnitude of the signal thresholds. 
         * @param nonDecisionTime Amount of time in milliseconds in which only noise 
         * is added to the decision variable. 
         * @param bias Corresponds to the initial value of the decision variable. Must 
         * be smaller than barrier. 
         */
        aDDM(
            float d, float sigma, float theta, float barrier, 
            unsigned int nonDecisionTime=0, float bias=0
        );


        /**
         * @brief Construct an empty aDDM object. 
         * 
         */
        aDDM() {}


        /**
         * @brief Compute the likelihood of the trial results provided the current parameters.
         * 
         * @param trial aDDMTrial object.
         * @param debug Boolean sepcifying if state variables should be printed for debugging
         * purposes.
         * @param timeStep Value in milliseconds used for binning the time axis.
         * @param approxstateStep Used for binning the RDV axis.
         * @return double representing the likelihood for the given trial. 
         */
        double getTrialLikelihood(aDDMTrial trial, bool debug=false, 
            int timeStep=10, float approxStateStep=0.1);


        /**
         * @brief Generate simulated fixations provided item values and empirical fixation data. 
         * 
         * @param valueLeft value of the left item
         * @param valueRight value of the right item
         * @param fixationData instance of a FixationData object containing empirical fixation data
         * @param timeStep value of in milliseconds used for binning time axis. 
         * @param numFixDists number of expected fixations in a given trial 
         * @param fixationDist distribution of the fixation data being used. 
         * @param timeBins predetermined time bins as used in the fixationDist. 
         * @param seed used for standardizing any random number generators. 
         * @return aDDMTrial resulting from the simulation. 
         */
        aDDMTrial simulateTrial(
            int valueLeft, int valueRight, FixationData fixationData, int timeStep=10, 
            int numFixDists=3, fixDists fixationDist={}, vector<int> timeBins={}, int seed=-1
        );


        /**
         * @brief Compute the total Negative Log Likelihood (NLL) for a vector of aDDMTrials. Use CPU
         * multithreading to maximize the number of blocks of trials that can have their respective 
         * NLLs computed in parallel. 
         * 
         * @param trials Vector of aDDMTrials that the model should calculcate the NLL for. 
         * @param debug Boolean specifying if state variables should be printed for debugging purposes.
         * @param timeStep Value in milliseconds used for binning the time axis. 
         * @param approxStateStep Used for binning the RDV axis.
         * @return ProbabilityData containing NLL, sum of likelihoods, and list of all computed 
         * likelihoods. 
         */
        ProbabilityData computeParallelNLL(
            vector<aDDMTrial> trials, bool debug=false, int timeStep=10, 
            float approxStateStep=0.1
        );


#ifndef EXCLUDE_CUDA_CODE
        /**
         * @brief Compute the total Negative Log Likelihood (NLL) for a vector of aDDMTrials. Use the
         * GPU to maximize the number of trials being computed in parallel. 
         * 
         * @param trials Vector of aDDMTrials that the model should calculcate the NLL for. 
         * @param debug Boolean specifying if state variables should be printed for debugging purposes.
         * @param trialsPerThread Number of trials that each thread should be designated to compute. 
         * Must be divisible by the total number of trials. 
         * @param timeStep Value in milliseconds used for binning the time axis. 
         * @param approxStateStep Used for binning the RDV axis.
         * @return ProbabilityData containing NLL, sum of likelihoods, and list of all computed 
         * likelihoods.
         */
        ProbabilityData computeGPUNLL(
            vector<aDDMTrial> trials, bool debug=false, int trialsPerThread=10, 
            int timeStep=10, float approxStateStep=0.1
        );
#endif 


        /**
         * @brief Complete a grid-search based Maximum Likelihood Estimation of all possible parameter 
         * combinations (d, theta, sigma) to determine which parameters are most likely to generate 
         * the provided aDDMTrials. Each potential model generates an NLL value for the dataset and the
         * method returns the model with the minimum NLL value. 
         * 
         * @param trials Vector of aDDMTrials that each model should calculate the NLL for. 
         * @param rangeD Vector of floats representing possible values of d to test for. 
         * @param rangeSigma Vector of floats representing possible values of sigma to test for. 
         * @param rangeTheta Vector of floats representing possible values of theta to test for. 
         * @param barrier Positive magnitude of the signal threshold. 
         * @param computeMethod Computation method to calculate the NLL for each possible model. 
         * Allowed values are {basic, thread, gpu}. "basic" will compute each trial likelihood 
         * sequentially and compute the NLL as the sum of all negative log likelihoods. "thread" will
         * use a thread pool to divide all trials into the maximum number of CPU threads and compute
         * the NLL of each block in parallel. "gpu" will call a CUDA kernel to compute the likelihood
         * of each trial in parallel on the GPU. 
         * @param normalizePosteriors true if the returned MLEinfo should contain a mapping of aDDMs 
         * to the normzlied posteriors distribution for each model; otherwise, the MLEinfo should 
         * containing a mapping of aDDMs to its corresponding NLL. 
         * @return MLEinfo containing the most optimal model and a mapping of models to floats 
         * determined by the normalizePosteriors argument. 
         */
        static MLEinfo<aDDM> fitModelMLE(
            vector<aDDMTrial> trials, vector<float> rangeD, vector<float> rangeSigma, 
            vector<float> rangeTheta, float barrier, string computeMethod="basic", 
            bool normalizePosteriors=false
        );
};

#endif 