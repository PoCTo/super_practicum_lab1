#pragma comment(lib, "mpi.lib")
#pragma comment(lib, "cxx.lib")

#define REDUCE_NODE 0 

#include <iostream>
#include <fstream>

#include <random>
#include <string>
#include <algorithm>
#include <ctime>
#include <cmath>
#include <cstring>

#include <mpi.h>
#include <omp.h>

// precounted quantiles
const double bins[99] = {
	-2.326, -2.278531, -2.231061, -2.183592, -2.136122, -2.088653, -2.041184, -1.993714, -1.946245, -1.898776, -1.851306, -1.803837, -1.756367, -1.708898, -1.661429, -1.613959, -1.56649, -1.51902, -1.471551, -1.424082, -1.376612, -1.329143, -1.281673, -1.234204, -1.186735, -1.139265, -1.091796, -1.044327, -0.9968571, -0.9493878, -0.9019184, -0.854449, -0.8069796, -0.7595102, -0.7120408, -0.6645714, -0.617102, -0.5696327, -0.5221633, -0.4746939, -0.4272245, -0.3797551, -0.3322857, -0.2848163, -0.2373469, -0.1898776, -0.1424082, -0.09493878, -0.04746939, 0, 0.04746939, 0.09493878, 0.1424082, 0.1898776, 0.2373469, 0.2848163, 0.3322857, 0.3797551, 0.4272245, 0.4746939, 0.5221633, 0.5696327, 0.617102, 0.6645714, 0.7120408, 0.7595102, 0.8069796, 0.854449, 0.9019184, 0.9493878, 0.9968571, 1.044327, 1.091796, 1.139265, 1.186735, 1.234204, 1.281673, 1.329143, 1.376612, 1.424082, 1.471551, 1.51902, 1.56649, 1.613959, 1.661429, 1.708898, 1.756367, 1.803837, 1.851306, 1.898776, 1.946245, 1.993714, 2.041184, 2.088653, 2.136122, 2.183592, 2.231061, 2.278531, 2.326
};

using std::cout;
using std::endl;

int main(int argc, char** argv) {
	double globalsum;
	int generateByNode;
	double startWtime = 0;
	double endWtime;
	int globalBinSizes[100];

	int nOMP = std::atoi(argv[1]);
	int nNorm = std::atoi(argv[2]);

	MPI::Init(argc, argv);

	int numProcs = MPI::COMM_WORLD.Get_size();
	int nodeId = MPI::COMM_WORLD.Get_rank();

	omp_set_dynamic(0);

#ifdef _DEBUG
	int namelen;
	char processor_name[MPI_MAX_PROCESSOR_NAME];
	MPI::Get_processor_name(processor_name, namelen);
	cout << "Initialized MPI worker " << nodeId << " of " << numProcs << " on " <<
		processor_name << endl;
#endif

	if (nodeId == REDUCE_NODE) {
		startWtime = MPI::Wtime();
	}

	MPI::COMM_WORLD.Bcast(&nOMP, 1, MPI::INT, REDUCE_NODE);
	MPI::COMM_WORLD.Bcast(&nNorm, 1, MPI::INT, REDUCE_NODE);

	omp_set_num_threads(nOMP);

	generateByNode = nNorm / numProcs;
	
	if (nodeId == REDUCE_NODE) {
		generateByNode += nNorm % numProcs;
	}	

	int *histByThread;
	double sum = 0;
	
    int binSizes[100] = {0};			
	histByThread = new int[100 * nOMP];

#pragma omp parallel 
{
    const int ithread = omp_get_thread_num();

	std::default_random_engine generator((unsigned int)(std::time(NULL)) ^ (ithread + nodeId * nOMP + 1));
	std::normal_distribution<double> distribution(0, 1);

    #pragma omp for
    for (int i = 0; i < 100 * nOMP; ++i) {
        histByThread[i] = 0;
    }

	#pragma omp for reduction(+:sum)
	for (int i = 0; i < generateByNode; ++i) {				
		double val = distribution(generator);
		sum += val;
		const double* upperBound = std::upper_bound(bins, bins + 99, val);
		histByThread[(upperBound - bins) + 100 * ithread]++;
	}

	#pragma omp for
	for (int i = 0; i < 100; ++i) {
        binSizes[i] = 0;
		for (int j = 0; j < nOMP; ++j) {            
			binSizes[i] += histByThread[i + 100 * j];
		}
	}
}	

	delete[] histByThread;

	MPI::COMM_WORLD.Reduce(&sum, &globalsum, 1, MPI::DOUBLE, MPI::SUM, REDUCE_NODE);
	MPI::COMM_WORLD.Reduce(binSizes, globalBinSizes, 100, MPI::INT, MPI::SUM, REDUCE_NODE);

#ifdef _DEBUG
	cout << "MPI worker " << nodeId << " of " << numProcs << " generated " <<
		generateByNode << " numbers and got sum = " << sum << endl;

	if (nodeId == REDUCE_NODE) {
		endWtime = MPI::Wtime();
		cout << "Time elapsed: " << (endWtime - startWtime) << "sec" << endl;
		cout << "Master node " << nodeId << " returned overall sum = " << globalsum << endl;
	}
#else 
	if (nodeId == REDUCE_NODE) {
		endWtime = MPI::Wtime();
		cout << (endWtime - startWtime) << " " << globalsum;
	}
#endif
	
	if (nodeId == REDUCE_NODE) {
		std::ofstream histCSV("hist.csv");
		histCSV << "-Inf" << "," << bins[0] << "," << globalBinSizes[0] << endl;
		for (int i = 0; i < 98; ++i) {
			histCSV << bins[i] << "," << bins[i + 1] << "," << globalBinSizes[i + 1] << endl;
		}
		histCSV << bins[98] << "," << "Inf" << "," << globalBinSizes[99] << endl;
	}

	MPI::Finalize();
	return 0;
}