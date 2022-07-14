#include "FpjClass_Agent.cuh"
#include <stdio.h>
#include "stdafx.h"

#define PI 3.1415926536f
#define STEPSIZE 0.2f

__global__ void InitDistance(float *distance_array, const float distance, const int V)
{
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	if (tid < V)
	{
		distance_array[tid] = distance;
	}
}

__global__ void InitU(float* u, const int N, const float du, const float offcenter)
{
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	if (tid < N)
	{
		u[tid] = (tid - (N - 1) / 2.0f) * du + offcenter;
	}
}

__global__ void InitBeta(float* beta, const int V, const float startAngle, const float totalScanAngle)
{
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	if (tid<V)
	{
		beta[tid] = (totalScanAngle / V * tid + startAngle) * PI / 180.0f;
	}
}

__global__ void PMatrixInv3_device(float *pmatrix)
{
    __shared__ float PMatrixInv3[9];

    int isx = threadIdx.x;
    int isy = threadIdx.y;
    float tmpIn;
    float tmpInv;
    // initialize E
    if(isx == isy)
        PMatrixInv3[isy*3 + isx] = 1;
    else
        PMatrixInv3[isy*3 + isx] = 0;

    // Gaussian elimination method for matrix inverse
    for (int i = 0; i < 3; i++)
    {
        if (i == isy && isx < 3 && isy < 3)
        {
            // The main diagonal element is reduced to 1
            tmpIn = pmatrix[i*3 + i];
            pmatrix[i*3 + isx] /= tmpIn;
            PMatrixInv3[i*3 + isx] /= tmpIn;
        }
        __syncthreads();
        if (i != isy && isx < 3 && isy < 3)
        {
            // Reduce the element in the pivot column to 0, and the element in the row changes simultaneously
            tmpInv = pmatrix[isy*3 + i];
            pmatrix[isy*3 + isx] -= tmpInv * pmatrix[i*3 + isx];
            PMatrixInv3[isy*3 + isx] -= tmpInv * PMatrixInv3[i*3 + isx];
        }
        __syncthreads();
    }

    pmatrix[isy*3 + isx] = PMatrixInv3[isy*3 + isx];
}

// img: image data
// sgm: sinogram data
// u: array of each detector element position
// beta: array of each view angle [radian]
// M: image dimension
// S: number of image slices
// N_z: number of detector elements in Z direction
// N: number of detector elements (sinogram width)
// V: number of views (sinogram height)
// dx: image pixel size [mm]
// dz: image slice thickness [mm]
// sid: source to isocenter distance
// sdd: source to detector distance
__global__ void ForwardProjectionBilinear_device(float* img, float* sgm, const float* u, const float *v, const float* offcenter_array, const float* beta, const float* swing_angle_array, int M, int S,\
	int N, int N_z, int V, float dx, float dz,  const float* sid_array, const float* sdd_array, bool conebeam, bool helican_scan, float helical_pitch,\
	int z_element_begin_idx, int z_element_end_idx)
{
	int col = threadIdx.x + blockDim.x * blockIdx.x;//column is direction of elements
	int row = threadIdx.y + blockDim.y * blockIdx.y;//row is direction of views
	//function is parallelly run for each element in each view

	if (col < N && row < V && z_element_end_idx <= N_z)
	{
		// half of image side length
		float D = float(M)  * dx / 2.0f;
		// half of image thickness
		float D_z = 0.0f;
		if (conebeam)
		{
			D_z = float(S) * dz / 2.0f;
		}
		else
		{
			dz = 0;
		}


		//get the sid and sdd for a given view
		float sid = sid_array[row];
		float sdd = sdd_array[row];

		// current source position
		float xs = sid * cosf(beta[row]);
		float ys = sid * sinf(beta[row]);
		float zs = 0;

		// calculate offcenter bias
		float offcenter_bias = offcenter_array[row] - offcenter_array[0];

		// current detector element position
		float xd = -(sdd - sid) * cosf(beta[row]) + (u[col]+ offcenter_bias) * cosf(beta[row] - PI/2.0f + swing_angle_array[row] /180.0f*PI);
		float yd = -(sdd - sid) * sinf(beta[row]) + (u[col]+ offcenter_bias) * sinf(beta[row] - PI/2.0f + swing_angle_array[row] / 180.0f*PI);
		float zd = 0;

		// step point region
		float L_min = sid - sqrt(2 * D * D + D_z * D_z);
		float L_max = sid + sqrt(2 * D * D + D_z * D_z);

		// source to detector element distance
		float sed = sqrtf((xs - xd)*(xs - xd) + (ys - yd)*(ys - yd));// for fan beam case

		// the point position
		float x, y, z;
		// the point index
		int kx, ky, kz;
		// weighting factor for linear interpolation
		float wx, wy, wz;

		// the most upper left image pixel position
		float x0 = -D + dx / 2.0f;
		float y0 = D - dx / 2.0f;
		float z0 = 0;
		if (conebeam)
		{
			z0 = -D_z + dz / 2.0f;// first slice is at the bottom, coordinate is -D_z +dz/2
			// last slice is at the top, coordinate is D_z -dz/2
		}

		float z_dis_per_view = 0;
		if (helican_scan)// for helical scan, we need to calculate the distance of the movement along the z direction between views
		{	
			float total_scan_angle = abs((beta[V - 1] - beta[0])) / float(V - 1)*float(V);
			float num_laps = total_scan_angle / (PI *2);
			z_dis_per_view = helical_pitch * (num_laps / V) * (abs(v[1]-v[0])*N_z) / (sdd / sid);
			//distance moved per view is pitch * lap per view * detector height / magnification

		}

		// repeat for each slice
		for (int slice = z_element_begin_idx; slice < z_element_end_idx; slice++)
		{
			// initialization
			//sgm[row*N + col + N * V * slice] = 0;
			sgm[row*N + col] = 0;
			if (conebeam)
			{
				zd = v[slice];
				sed = sqrtf((xs - xd)*(xs - xd) + (ys - yd)*(ys - yd) + (zs - zd)*(zs - zd));
			}

			// calculate line integration
			for (float L = L_min; L <= L_max; L+= STEPSIZE*sqrt(dx*dx+dz*dz/2.0f))
			// for (float L = L_min; L <= L_max; L+= STEPSIZE*dx)  // <=
			{
				// ratio of [distance: current position to source] to [distance: source to element]
				float ratio_L_sed = L / sed;

				// get the current point position 
				x = xs + (xd - xs) * ratio_L_sed;
				y = ys + (yd - ys) * ratio_L_sed;
				if (conebeam)// for cone beam, we need to calculate the z position
				{
					z = zs + (zd - zs) * ratio_L_sed;
				}

				if (helican_scan)
				{
					z = z + z0 + row * z_dis_per_view;
					//for helical scan, if the image object is treated as stationary, both the detector and the source should move upward
				}

				// get the current point index
				kx = floorf((x - x0) / dx);
				ky = floorf((y0 - y) / dx);

				if (conebeam)
					kz = floorf((z - z0) / dz);
					// kz = roundf((z - z0) / dz);// floorf((z - z0) / dz);  // <=

				// get the image pixel value at the current point
				if(kx>=0 && kx+1<M && ky>=0 && ky+1<M)
				{
					// get the weighting factor
					wx = (x - kx * dx - x0) / dx;
					wy = (y0 - y - ky * dx) / dx;

					// perform bilinear interpolation
					if (conebeam == false)
					{
						sgm[row*N + col] += (1 - wx)*(1 - wy)*img[ky*M + kx + M * M*slice] // upper left
							+ wx * (1 - wy) * img[ky*M + kx + 1 + M * M*slice] // upper right
							+ (1 - wx) * wy * img[(ky + 1)*M + kx + M * M*slice] // bottom left
							+ wx * wy * img[(ky + 1)*M + kx + 1 + M * M*slice];	// bottom right
					}
					else if (conebeam == true && kz >= 0 && kz + 1 < S)
					{

						wz = (z - kz * dz - z0) / dz;
						float sgm_val_lowerslice = (1 - wx)*(1 - wy)*img[ky*M + kx + M * M*kz] // upper left
							+ wx * (1 - wy) * img[ky*M + kx + 1 + M * M*kz] // upper right
							+ (1 - wx) * wy * img[(ky + 1)*M + kx + M * M*kz] // bottom left
							+ wx * wy * img[(ky + 1)*M + kx + 1 + M * M*kz];	// bottom right
						float sgm_val_upperslice = (1 - wx)*(1 - wy)*img[ky*M + kx + M * M*(kz+1)] // upper left
							+ wx * (1 - wy) * img[ky*M + kx + 1 + M * M*(kz + 1)] // upper right
							+ (1 - wx) * wy * img[(ky + 1)*M + kx + M * M*(kz + 1)] // bottom left
							+ wx * wy * img[(ky + 1)*M + kx + 1 + M * M*(kz + 1)];	// bottom right

						sgm[row*N + col] += (1 - wz)*sgm_val_lowerslice + wz * sgm_val_upperslice;

						/*  <=
						float sgm_val_temp = (1 - wx)*(1 - wy)*img[ky*M + kx + M * M*kz] // upper left
							+ wx * (1 - wy) * img[ky*M + kx + 1 + M * M*kz] // upper right
							+ (1 - wx) * wy * img[(ky + 1)*M + kx + M * M*kz] // bottom left
							+ wx * wy * img[(ky + 1)*M + kx + 1 + M * M*kz];	// bottom right

						sgm[row*N + col] += sgm_val_temp;//(1 - wz)*sgm_val_lowerslice + wz * sgm_val_upperslice;
						*/
					}

				}
			}
			sgm[row*N + col] *= STEPSIZE * sqrt(dx*dx + dz * dz/2.0f);
			// sgm[row*N + col] *= STEPSIZE * dx;  // <=

		}
	}
}

// img: image data
// sgm: sinogram data
// u: array of each detector element position
// beta: array of each view angle [radian]
// M: image dimension
// S: number of image slices
// N_z: number of detector elements in Z direction
// N: number of detector elements (sinogram width)
// V: number of views (sinogram height)
// dx: image pixel size [mm]
// dz: image slice thickness [mm]
// sid: source to isocenter distance
// sdd: source to detector distance
__global__ void ForwardProjectionBilinear_pmatrix_device(float* img, float* sgm, const float* u, const float *v, const float* pmatrix, const float* beta, const float* swing_angle_array, int M, int S,\
	int N, int N_z, int V, float dx, float dz,  const float* sid_array, const float* sdd_array, bool conebeam, bool helican_scan, float helical_pitch,\
	int z_element_begin_idx, int z_element_end_idx, int osSize)
{
    int col = threadIdx.x + blockDim.x * blockIdx.x;//column is direction of elements
    int row = threadIdx.y + blockDim.y * blockIdx.y;//row is direction of views
    //function is parallelly run for each element in each view

    if (col < N && row < V && z_element_end_idx <= N_z)
    {
        // half of image side length
        float D = float(M)  * dx / 2.0f;
        // half of image thickness
        float D_z = 0.0f;
        if (conebeam)
        {
            D_z = float(S) * dz / 2.0f;
        }
        else
        {
            dz = 0;
        }


        // get the sid and sdd for a given view
        float sid = sid_array[row];  // now useless for cone beam and fan beam
        float sdd = sdd_array[row];  // now useless for cone beam and fan beam

        // pmatrix index and params
        int pos_in_matrix = 12 * row;
        float p_14 = pmatrix[pos_in_matrix + 3];
        float p_24 = pmatrix[pos_in_matrix + 7];
        float p_34 = pmatrix[pos_in_matrix + 11];

        // current source position
        float xs = pmatrix[pos_in_matrix + 0] * -p_14 \
                 + pmatrix[pos_in_matrix + 1] * -p_24 \
                 + pmatrix[pos_in_matrix + 2] * -p_34;

        float ys = pmatrix[pos_in_matrix + 4] * -p_14 \
                 + pmatrix[pos_in_matrix + 5] * -p_24 \
                 + pmatrix[pos_in_matrix + 6] * -p_34;

        float zs = pmatrix[pos_in_matrix + 8] * -p_14 \
                 + pmatrix[pos_in_matrix + 9] * -p_24 \
                 + pmatrix[pos_in_matrix + 10]* -p_34;
        // reset SID from source position
        sid = sqrtf(xs * xs + ys * ys);

        // current detector element position
        float xd = pmatrix[pos_in_matrix + 0] * (1 * ((col + 0.5f) / float(osSize) - 0.5f) - p_14) \
                 + pmatrix[pos_in_matrix + 1] * (1 * z_element_begin_idx - p_24) \
                 + pmatrix[pos_in_matrix + 2] * (1 - p_34);

        float yd = pmatrix[pos_in_matrix + 4] * (1 * ((col + 0.5f) / float(osSize) - 0.5f) - p_14) \
                 + pmatrix[pos_in_matrix + 5] * (1 * z_element_begin_idx - p_24) \
                 + pmatrix[pos_in_matrix + 6] * (1 - p_34);

        float zd = 0;

        // step point region
        float L_min = sid - sqrt(2 * D * D + D_z * D_z);
        float L_max = sid + sqrt(2 * D * D + D_z * D_z);

        // source to detector element distance
        float sed = sqrtf((xs - xd)*(xs - xd) + (ys - yd)*(ys - yd));// for fan beam case

        // the point position
        float x, y, z;
        // the point index
        int kx, ky, kz;
        // weighting factor for linear interpolation
        float wx, wy, wz;

        // the most upper left image pixel position
        float x0 = -D + dx / 2.0f;
        float y0 = D - dx / 2.0f;
        float z0 = 0;
        if (conebeam)
        {
            z0 = -D_z + dz / 2.0f;// first slice is at the bottom, coordinate is -D_z +dz/2
            // last slice is at the top, coordinate is D_z -dz/2
        }

        float z_dis_per_view = 0;
        if (helican_scan)// for helical scan, we need to calculate the distance of the movement along the z direction between views
        {
            float total_scan_angle = abs((beta[V - 1] - beta[0])) / float(V - 1)*float(V);
            float num_laps = total_scan_angle / (PI *2);
            z_dis_per_view = helical_pitch * (num_laps / V) * (abs(v[1]-v[0])*N_z) / (sdd / sid);
            //distance moved per view is pitch * lap per view * detector height / magnification

        }

        // repeat for each slice
        for (int slice = z_element_begin_idx; slice < z_element_end_idx; slice++)
        {
            // initialization
            //sgm[row*N + col + N * V * slice] = 0;
            sgm[row*N + col] = 0;
            if (conebeam)
            {
                zd = pmatrix[pos_in_matrix + 8] * (1 * ((col + 0.5f) / float(osSize) - 0.5f) - p_14) \
                   + pmatrix[pos_in_matrix + 9] * (1 * slice - p_24) \
                   + pmatrix[pos_in_matrix + 10]* (1 - p_34);
                sed = sqrtf((xs - xd)*(xs - xd) + (ys - yd)*(ys - yd) + (zs - zd)*(zs - zd));
            }

            // calculate line integration
            for (float L = L_min; L <= L_max; L+= STEPSIZE*sqrt(dx*dx+dz*dz/2.0f))
                // for (float L = L_min; L <= L_max; L+= STEPSIZE*dx)  // <=
            {
                // ratio of [distance: current position to source] to [distance: source to element]
                float ratio_L_sed = L / sed;

                // get the current point position
                x = xs + (xd - xs) * ratio_L_sed;
                y = ys + (yd - ys) * ratio_L_sed;

                if (conebeam)// for cone beam, we need to calculate the z position
                {
                    z = zs + (zd - zs) * ratio_L_sed;
                }

                if (helican_scan)
                {
                    z = z + z0 + row * z_dis_per_view;
                    //for helical scan, if the image object is treated as stationary, both the detector and the source should move upward
                }

                // get the current point index
                kx = floorf((x - x0) / dx);
                ky = floorf((y0 - y) / dx);

                if (conebeam)
                    kz = floorf((z - z0) / dz);
                // kz = roundf((z - z0) / dz);// floorf((z - z0) / dz);  // <=

                // get the image pixel value at the current point
                if(kx>=0 && kx+1<M && ky>=0 && ky+1<M)
                {
                    // get the weighting factor
                    wx = (x - kx * dx - x0) / dx;
                    wy = (y0 - y - ky * dx) / dx;

                    // perform bilinear interpolation
                    if (conebeam == false)
                    {
                        sgm[row*N + col] += (1 - wx)*(1 - wy)*img[ky*M + kx + M * M*slice] // upper left
                                            + wx * (1 - wy) * img[ky*M + kx + 1 + M * M*slice] // upper right
                                            + (1 - wx) * wy * img[(ky + 1)*M + kx + M * M*slice] // bottom left
                                            + wx * wy * img[(ky + 1)*M + kx + 1 + M * M*slice];	// bottom right
                    }
                    else if (conebeam == true && kz >= 0 && kz + 1 < S)
                    {

                        wz = (z - kz * dz - z0) / dz;
                        float sgm_val_lowerslice = (1 - wx)*(1 - wy)*img[ky*M + kx + M * M*kz] // upper left
                                                   + wx * (1 - wy) * img[ky*M + kx + 1 + M * M*kz] // upper right
                                                   + (1 - wx) * wy * img[(ky + 1)*M + kx + M * M*kz] // bottom left
                                                   + wx * wy * img[(ky + 1)*M + kx + 1 + M * M*kz];	// bottom right
                        float sgm_val_upperslice = (1 - wx)*(1 - wy)*img[ky*M + kx + M * M*(kz+1)] // upper left
                                                   + wx * (1 - wy) * img[ky*M + kx + 1 + M * M*(kz + 1)] // upper right
                                                   + (1 - wx) * wy * img[(ky + 1)*M + kx + M * M*(kz + 1)] // bottom left
                                                   + wx * wy * img[(ky + 1)*M + kx + 1 + M * M*(kz + 1)];	// bottom right

                        sgm[row*N + col] += (1 - wz)*sgm_val_lowerslice + wz * sgm_val_upperslice;

                    }

                }
            }
            sgm[row*N + col] *= STEPSIZE * sqrt(dx*dx + dz * dz/2.0f);

        }
    }
}

// sgm_large: sinogram data before binning
// sgm: sinogram data after binning
// N: number of detector elements (after binning)
// V: number of views
// S: number of slices
// binSize: bin size
__global__ void BinSinogram(float* sgm_large, float* sgm, int N, int V, int S, int binSize)
{
	int col = threadIdx.x + blockDim.x * blockIdx.x;
	int row = threadIdx.y + blockDim.y * blockIdx.y;

	if (col < N && row < V)
	{
		// repeat for each slice
		for (int slice = 0; slice < S; slice++)
		{
			// initialization
			sgm[row * N + col + N * V * slice] = 0;

			// sum over each bin
			for (int i = 0; i < binSize; i++)
			{
				sgm[row * N + col + N * V * slice] += sgm_large[row * N * binSize + col*binSize + i + slice * N * binSize * V];
			}
			// take average
			sgm[row * N + col + N * V * slice] /= binSize;
		}
	}
}

void InitializeDistance_Agent(float* &distance_array, const float distance, const int V)
{
	if (distance_array != nullptr)
		cudaFree(distance_array);

	cudaMalloc((void**)&distance_array, V * sizeof(float));
	InitDistance << <(V + 511) / 512, 512 >> > (distance_array, distance, V);
}

void InitializeNonuniformSDD_Agent(float* &distance_array, const int V, const std::string& distanceFile)
{
	// namespace fs = std::filesystem;
	namespace js = rapidjson;

	if (distance_array != nullptr)
		cudaFree(distance_array);

	cudaMalloc((void**)&distance_array, V * sizeof(float));

	float* distance_array_cpu = new float[V];
	std::ifstream ifs(distanceFile);
	if (!ifs)
	{
		printf("Cannot find SDD information file '%s'!\n", distanceFile.c_str());
		exit(-2);
	}
	rapidjson::IStreamWrapper isw(ifs);
	rapidjson::Document doc;
	doc.ParseStream<js::kParseCommentsFlag | js::kParseTrailingCommasFlag>(isw);
	js::Value distance_jsonc_value;
	if (doc.HasMember("SourceDetectorDistance"))
	{

		distance_jsonc_value = doc["SourceDetectorDistance"];

		if (distance_jsonc_value.Size() != V)
		{
			printf("Number of sdd values is %d while the number of Views is %d!\n", distance_jsonc_value.Size(), V);
			exit(-2);
		}

		for (unsigned i = 0; i < distance_jsonc_value.Size(); i++)
		{
			distance_array_cpu[i] = distance_jsonc_value[i].GetFloat();
		}

	}
	else
	{
		printf("Did not find SourceDetectorDistance member in jsonc file!\n");
		exit(-2);
	}

	cudaMemcpy(distance_array, distance_array_cpu, sizeof(float)*V, cudaMemcpyHostToDevice);

	cudaDeviceSynchronize();
}

void InitializeNonuniformSID_Agent(float* &distance_array, const int V, const std::string& distanceFile)
{
	// namespace fs = std::filesystem;
	namespace js = rapidjson;

	if (distance_array != nullptr)
		cudaFree(distance_array);

	cudaMallocManaged((void**)&distance_array, V * sizeof(float));
	std::ifstream ifs(distanceFile);
	if (!ifs)
	{
		printf("Cannot find SID information file '%s'!\n", distanceFile.c_str());
		exit(-2);
	}
	rapidjson::IStreamWrapper isw(ifs);
	rapidjson::Document doc;
	doc.ParseStream<js::kParseCommentsFlag | js::kParseTrailingCommasFlag>(isw);
	js::Value distance_jsonc_value;
	if (doc.HasMember("SourceIsocenterDistance"))
	{

		distance_jsonc_value = doc["SourceIsocenterDistance"];

		if (distance_jsonc_value.Size() != V)
		{
			printf("Number of sid values is %d while the number of Views is %d!\n", distance_jsonc_value.Size(), V);
			exit(-2);
		}

		for (unsigned i = 0; i < distance_jsonc_value.Size(); i++)
		{
			distance_array[i] = distance_jsonc_value[i].GetFloat();
		}

	}
	else
	{
		printf("Did not find SourceIsocenterDistance member in jsonc file!\n");
		exit(-2);
	}

	cudaDeviceSynchronize();
}

void InitializeNonuniformOffCenter_Agent(float* &offcenter_array, const int V, const std::string& offCenterFile)
{
	// namespace fs = std::filesystem;
	namespace js = rapidjson;

	if (offcenter_array != nullptr)
		cudaFree(offcenter_array);

	cudaMallocManaged((void**)&offcenter_array, V * sizeof(float));
	std::ifstream ifs(offCenterFile);
	if (!ifs)
	{
		printf("Cannot find Offcenter or Swing Angle information file '%s'!\n", offCenterFile.c_str());
		exit(-2);
	}
	rapidjson::IStreamWrapper isw(ifs);
	rapidjson::Document doc;
	doc.ParseStream<js::kParseCommentsFlag | js::kParseTrailingCommasFlag>(isw);
	js::Value distance_jsonc_value;
	if (doc.HasMember("OffcenterArray"))
	{

		distance_jsonc_value = doc["OffcenterArray"];

		if (distance_jsonc_value.Size() != V)
		{
			printf("Number of offcenter values is %d while the number of Views is %d!\n", distance_jsonc_value.Size(), V);
			exit(-2);
		}

		for (unsigned i = 0; i < distance_jsonc_value.Size(); i++)
		{
			offcenter_array[i] = distance_jsonc_value[i].GetFloat();
		}

	}
	else
	{
		printf("Did not find OffcenterArray member in jsonc file!\n");
		exit(-2);
	}

	cudaDeviceSynchronize();
}

//new function with Value member to suit all non uniform parameters
void InitializeNonuniformPara_Agent(float* &para_array, const int V, const std::string& paraFile)
{
	// namespace fs = std::filesystem;
	namespace js = rapidjson;

	if (para_array != nullptr)
		cudaFree(para_array);

	cudaMalloc((void**)&para_array, V * sizeof(float));
	float* para_array_cpu = new float[V];

	std::ifstream ifs(paraFile);
	if (!ifs)
	{
		printf("Cannot find file '%s'!\n", paraFile.c_str());
		exit(-2);
	}
	rapidjson::IStreamWrapper isw(ifs);
	rapidjson::Document doc;
	doc.ParseStream<js::kParseCommentsFlag | js::kParseTrailingCommasFlag>(isw);
	js::Value array_jsonc_value;
	if (doc.HasMember("Value"))
	{
		array_jsonc_value = doc["Value"];
	}
	else
	{
		printf("Did not find Value member in jsonc file!\n");
		exit(-2);
	}


	if (array_jsonc_value.Size() != V)
	{
		printf("Number of elements in the array is %d while the number of Views is %d!\n", array_jsonc_value.Size(), V);
		exit(-2);
	}

	for (unsigned i = 0; i < array_jsonc_value.Size(); i++)
	{
		para_array_cpu[i] = array_jsonc_value[i].GetFloat(); //printf("%d: %f\n", i, para_array_cpu[i]);
	}
	cudaMemcpy(para_array, para_array_cpu, sizeof(float)*V, cudaMemcpyHostToDevice);
	//printf("copy finished!\n");
	cudaDeviceSynchronize();
}

void InitializePMatrix_Agent(float* &pmatrix_array, const int V, const std::string& pmatrixFile)
{
    // namespace fs = std::filesystem;
    namespace js = rapidjson;

    if (pmatrix_array != nullptr)
        cudaFree(pmatrix_array);

    //cudaMallocManaged((void**)&pmatrix_array, 12 * V * sizeof(float));
    cudaMalloc((void**)&pmatrix_array, 12 * V * sizeof(float));
    //cudaMallocManaged somehow does not work for this function
    //so cudaMalloc and cudaMemcpy is used


    float* pmatrix_array_cpu = new float[12 * V];


    std::ifstream ifs(pmatrixFile);
    if (!ifs)
    {
        printf("\nCannot find pmatrix information file '%s'!\n", pmatrixFile.c_str());
        exit(-2);
    }
    rapidjson::IStreamWrapper isw(ifs);
    rapidjson::Document doc;
    doc.ParseStream<js::kParseCommentsFlag | js::kParseTrailingCommasFlag>(isw);
    js::Value pmatrix_jsonc_value;
    if (doc.HasMember("PMatrix"))
    {
        pmatrix_jsonc_value = doc["PMatrix"];
    }
    else if(doc.HasMember("Value"))
    {
        pmatrix_jsonc_value = doc["Value"];
    }
    else
    {
        printf("\nDid not find PMatrix or Value member in jsonc file!\n");
        exit(-2);
    }
    if (pmatrix_jsonc_value.Size() != 12 * V)
    {
        printf("\nNumber of pmatrix elements is %d while the 12 times number of Views is %d!\n", pmatrix_jsonc_value.Size(), 12 * V);
        exit(-2);
    }

    for (unsigned i = 0; i < 12 * V; i++)
    {
        //printf("\n%d: %f",i, pmatrix_jsonc_value[i].GetFloat());
        pmatrix_array_cpu[i] = pmatrix_jsonc_value[i].GetFloat();
    }
    cudaMemcpy(pmatrix_array, pmatrix_array_cpu, 12 * V * sizeof(float), cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();

    // Inverse pMatrix for the first three dimensions
    float* pmatrix = nullptr;
    cudaMalloc((void**)&pmatrix, 9 * sizeof(float));
    uint3 s;s.x = 3;s.y = 3;s.z = 1;

    for (unsigned i = 0; i < V; i++)
    {
        // Inverse pMatrix for the first three dimensions
        cudaMemcpy(&pmatrix[0], &pmatrix_array[12*i + 0], 3 * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(&pmatrix[3], &pmatrix_array[12*i + 4], 3 * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(&pmatrix[6], &pmatrix_array[12*i + 8], 3 * sizeof(float), cudaMemcpyDeviceToDevice);

        PMatrixInv3_device <<<1, s>>>(pmatrix);

        cudaMemcpy(&pmatrix_array[12*i + 0], &pmatrix[0], 3 * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(&pmatrix_array[12*i + 4], &pmatrix[3], 3 * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(&pmatrix_array[12*i + 8], &pmatrix[6], 3 * sizeof(float), cudaMemcpyDeviceToDevice);

    }
    cudaFree(pmatrix);
}

void InitializeU_Agent(float* &u, const int N, const float du, const float offcenter)
{
	if (u != nullptr)
		cudaFree(u);

	cudaMalloc((void**)&u, N * sizeof(float));
	InitU <<<(N + 511) / 512, 512 >>> (u, N, du, offcenter);
}

void InitializeBeta_Agent(float *& beta, const int V, const float startAngle, const float totalScanAngle)
{
	if (beta != nullptr)
		cudaFree(beta);

	cudaMalloc((void**)&beta, V * sizeof(float));
	InitBeta <<< (V + 511) / 512, 512 >>> (beta, V, startAngle, totalScanAngle);
}

void InitializeNonuniformBeta_Agent(float* &beta, const int V, const float rotation, const std::string& scanAngleFile)
{
	// namespace fs = std::filesystem;
	namespace js = rapidjson;

	if (beta != nullptr)
		cudaFree(beta);

	cudaMalloc((void**)&beta, V * sizeof(float));
	float * beta_cpu = new float[V];
	std::ifstream ifs(scanAngleFile);
	if (!ifs)
	{
		printf("Cannot find angle information file '%s'!\n", scanAngleFile.c_str());
		exit(-2);
	}
	rapidjson::IStreamWrapper isw(ifs);
	rapidjson::Document doc;
	doc.ParseStream<js::kParseCommentsFlag | js::kParseTrailingCommasFlag>(isw);
	js::Value scan_angle_jsonc_value;
	if (doc.HasMember("ScanAngle"))
	{
		scan_angle_jsonc_value = doc["ScanAngle"];
	}
	else if (doc.HasMember("Value"))
	{
		scan_angle_jsonc_value = doc["Value"];
	}
	else
	{
		printf("Did not find ScanAngle or Value member in jsonc file!\n");
		exit(-2);
	}

	if (scan_angle_jsonc_value.Size() != V)
	{
		printf("Number of scan angles is %d while the number of Views is %d!\n", scan_angle_jsonc_value.Size(), V);
		exit(-2);
	}

	for (unsigned i = 0; i < scan_angle_jsonc_value.Size(); i++)
	{
		beta_cpu[i] = rotation / 180.0f*PI + scan_angle_jsonc_value[i].GetFloat() / 180.0*PI;
	}
	cudaMemcpy(beta, beta_cpu, sizeof(float)*V, cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();
}

void ForwardProjectionBilinear_Agent(float *& image, float * &sinogram, const float* sid_array, const float* sdd_array, const float* offcenter_array,\
	const float* pmatrix_array, const float* u, const float *v, const float* beta, const float* swing_angle_array, const mango::Config & config, int z_element_idx)
{
	dim3 grid((config.detEltCount*config.oversampleSize + 7) / 8, (config.views + 7) / 8);
	dim3 block(8, 8);

	if (config.pmatrixFlag == false)// if pmatrix is not applied
    {
        ForwardProjectionBilinear_device<<<grid, block>>>(image, sinogram, u, v, offcenter_array, beta, swing_angle_array, config.imgDim, config.sliceCount,\
		config.detEltCount*config.oversampleSize, config.detZEltCount, config.views, config.pixelSize, config.sliceThickness,  sid_array, sdd_array, config.coneBeam, \
		config.helicalScan, config.helicalPitch,  z_element_idx, z_element_idx+1);
    }
	else if (config.pmatrixFlag == true)// if pmatrix is applied
    {
        ForwardProjectionBilinear_pmatrix_device<<<grid, block>>>(image, sinogram, u, v, pmatrix_array, beta, swing_angle_array, config.imgDim, config.sliceCount,\
		config.detEltCount*config.oversampleSize, config.detZEltCount, config.views, config.pixelSize, config.sliceThickness,  sid_array, sdd_array, config.coneBeam, \
		config.helicalScan, config.helicalPitch,  z_element_idx, z_element_idx+1, config.oversampleSize);
    }

	cudaDeviceSynchronize();
}

void BinSinogram(float* &sinogram_large, float* &sinogram, const mango::Config& config)
{
	dim3 grid((config.detEltCount + 7) / 8, (config.views + 7) / 8);
	dim3 block(8, 8);

	BinSinogram <<<grid, block >>> (sinogram_large, sinogram, config.detEltCount, config.views, 1, config.oversampleSize);
	// since the sinogram has only one slice, the z_element count is 1

	cudaDeviceSynchronize();
}

void SaveSinogramSlice(const char * filename, float*&sinogram_slice, int z_element_idx, const mango::Config& config)
{
	FILE* fp = NULL;
	if (z_element_idx == 0)
		fp = fopen(filename, "wb");
	else
		fp = fopen(filename, "ab");

	if (fp == NULL)
	{
		fprintf(stderr, "Cannot save to file %s!\n", filename);
		exit(4);
	}
	fwrite(sinogram_slice, sizeof(float), config.detEltCount * config.views, fp);
	fclose(fp);
}

void MallocManaged_Agent(float * &p, const int size)
{
	cudaMallocManaged((void**)&p, size);
}

void FreeMemory_Agent(float* &p)
{
	cudaFree(p);
	p = nullptr;
}
