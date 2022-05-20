// mgfbp.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "FbpClass.cuh"


int main(int argc, char* argv[])
{
	namespace mg = mango;
	// namespace fs = std::filesystem;

	if (argc == 1)
	{
		fprintf(stderr, "No config files!\n");
		return -1;
	}

	for (int idx = 1; idx < argc; idx++)
	{
		mg::FbpClass fbp;

		printf("Loading config %s...\n", argv[idx]);

		fbp.ReadConfigFile(argv[idx]);

		fbp.InitParam();

		fs::path inDir(mg::FbpClass::config.inputDir);
		fs::path outDir(mg::FbpClass::config.outputDir);

		// record time
        float time;
		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
        cudaEventRecord(start);

		for (size_t i = 0; i < mg::FbpClass::config.inputFiles.size(); i++)
		{
			printf("    Reconstructing %s ...", mg::FbpClass::config.inputFiles[i].c_str());

			fs::path sgmFile(mg::FbpClass::config.inputFiles[i]);

			fbp.ReadSinogramFile((inDir / sgmFile).string().c_str());

			if (mg::FbpClass::config.doBeamHardeningCorr)
			{
				fbp.CorrectBeamHardening();
			}

			fbp.FilterSinogram();

			if (mg::FbpClass::config.saveFilterSinogram)
			{
				fbp.SaveFilteredSinogram((outDir / ("filter_" + mg::FbpClass::config.inputFiles[i])).string().c_str());
			}

			fbp.BackprojectPixelDrivenAndSave((outDir / mg::FbpClass::config.outputFiles[i]).string().c_str());

			//fbp.SaveImage((outDir / mg::FbpClass::config.outputFiles[i]).string().c_str());

			printf("\n->\tSaved to file %s\n", mg::FbpClass::config.outputFiles[i].c_str());
		}

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&time, start, stop);
        printf("Total cost: %lf ms\n", time);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);

	}

	return 0;
}

