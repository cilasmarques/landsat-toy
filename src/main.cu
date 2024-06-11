#include <iostream>
#include <fstream>
#include <string>
#include <chrono>

#include "candidate.h"
#include "constants.h"
#include "cuda_utils.h"
#include "endmembers.h"
#include "kernels.cuh"

using namespace std;
using namespace std::chrono;

// Matrix environment
int NORMAL_HEIGHT_BAND = 35;  // int SMALL_HEIGHT_BAND = 35;
int NORMAL_WIDTH_BAND = 35;   // int SMALL_WIDTH_BAND = 35;

int height_band = NORMAL_HEIGHT_BAND;
int width_band = NORMAL_WIDTH_BAND;
int image_size = height_band * width_band;
int nBytes_band = height_band * width_band * sizeof(double);

// Global variables
Candidate hot_pixel;
Candidate cold_pixel;
double H_pf_terra;
double H_pq_terra;
double rah_ini_pq_terra;
double rah_ini_pf_terra;

double ndvi_min = 1.0;
double ndvi_max = -1.0;
double *ndvi_pointer = new double[image_size];
double *albedo_pointer = new double[image_size];
double *net_radiation_pointer = new double[image_size];
double *soil_heat_pointer = new double[image_size];
double *surface_pointer = new double[image_size];
double *zom_pointer = new double[image_size];
double *d0_pointer = new double[image_size];
double *kb1_pointer = new double[image_size];
double *ustar_pointer = new double[image_size];
double *aerodynamic_pointer = new double[image_size];
double *sensible_heat_pointer = new double[image_size];
vector<vector<double>> ndvi_vector;
vector<vector<double>> albedo_vector;
vector<vector<double>> surface_temperature_vector;
vector<vector<double>> net_radiation_vector;
vector<vector<double>> soil_heat_vector;

// Functions

string compute_initial_products()
{
  // Need to compute the initial products ( HERE WE ARE USING PRE COMPUTED VALUES )
  ifstream inFile;

  inFile.open("./input/small/NDVI.dat", std::ios::binary);
  inFile.read(reinterpret_cast<char*>(ndvi_pointer), image_size * sizeof(double));
  inFile.close();

  inFile.open("./input/small/ALBEDO.dat", std::ios::binary);
  inFile.read(reinterpret_cast<char *>(albedo_pointer), sizeof(double) * image_size);
  inFile.close();

  inFile.open("./input/small/NET_RADIATION.dat", std::ios::binary);
  inFile.read(reinterpret_cast<char *>(net_radiation_pointer), sizeof(double) * image_size);
  inFile.close();

  inFile.open("./input/small/SOIL_HEAT_FLUX.dat", std::ios::binary);
  inFile.read(reinterpret_cast<char *>(soil_heat_pointer), sizeof(double) * image_size);
  inFile.close();

  inFile.open("./input/small/TS.dat", std::ios::binary);
  inFile.read(reinterpret_cast<char *>(surface_pointer), sizeof(double) * image_size);
  inFile.close();

  // Need to add the values to the vectors
  for (int i = 0; i < height_band; i++)
  {
    vector<double> ndvi_row;
    vector<double> albedo_row;
    vector<double> surface_temperature_row;
    vector<double> net_radiation_row;
    vector<double> soil_heat_row;

    for (int j = 0; j < width_band; j++)
    {
      ndvi_row.push_back(ndvi_pointer[i * width_band + j]);
      albedo_row.push_back(albedo_pointer[i * width_band + j]);
      surface_temperature_row.push_back(surface_pointer[i * width_band + j]);
      net_radiation_row.push_back(net_radiation_pointer[i * width_band + j]);
      soil_heat_row.push_back(soil_heat_pointer[i * width_band + j]);
    }

    ndvi_vector.push_back(ndvi_row);
    albedo_vector.push_back(albedo_row);
    surface_temperature_vector.push_back(surface_temperature_row);
    net_radiation_vector.push_back(net_radiation_row);
    soil_heat_vector.push_back(soil_heat_row);
  }

  return "P2 - INITIAL PRODUCTS, 0, 0, 0\n";
};

string select_endmembers(int method)
{
  system_clock::time_point begin, end;
  int64_t general_time, initial_time, final_time;

  begin = system_clock::now();
  initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  if (method == 0)
  { // STEEP
    pair<Candidate, Candidate> pixels = getEndmembersSTEPP(ndvi_vector, surface_temperature_vector, albedo_vector, net_radiation_vector, soil_heat_vector, height_band, width_band);
    hot_pixel = pixels.first;
    cold_pixel = pixels.second;
  }
  else if (method == 1)
  { // ASEBAL
    pair<Candidate, Candidate> pixels = getEndmembersASEBAL(ndvi_vector, surface_temperature_vector, albedo_vector, net_radiation_vector, soil_heat_vector, height_band, width_band);
    hot_pixel = pixels.first;
    cold_pixel = pixels.second;
  }

  end = system_clock::now();
  general_time = duration_cast<nanoseconds>(end - begin).count();
  final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

  return "P2 - PIXEL SELECTION," + std::to_string(general_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
}

string compute_initial_rah()
{
  // Need to get the minimum and maximum NDVI values
  for (int i = 0; i < height_band; i++)
  {
    for (int j = 0; j < width_band; j++)
    {
      if (ndvi_vector[i][j] < ndvi_min)
        ndvi_min = ndvi_vector[i][j];
      if (ndvi_vector[i][j] > ndvi_max)
        ndvi_max = ndvi_vector[i][j];
    }
  }

  // Need to compute the initial aerodynamic resistance ( HERE WE ARE USING PRE COMPUTED VALUES )
  ifstream inFile;

  inFile.open("./input/small/ZOM.dat", std::ios::binary);
  inFile.read(reinterpret_cast<char *>(&image_size), sizeof(image_size));
  inFile.read(reinterpret_cast<char *>(zom_pointer), sizeof(int) * image_size);
  inFile.close();

  inFile.open("./input/small/D0.dat", std::ios::binary);
  inFile.read(reinterpret_cast<char *>(&image_size), sizeof(image_size));
  inFile.read(reinterpret_cast<char *>(d0_pointer), sizeof(int) * image_size);
  inFile.close();

  inFile.open("./input/small/KB1.dat", std::ios::binary);
  inFile.read(reinterpret_cast<char *>(&image_size), sizeof(image_size));
  inFile.read(reinterpret_cast<char *>(kb1_pointer), sizeof(int) * image_size);
  inFile.close();

  inFile.open("./input/small/USTAR.dat", std::ios::binary);
  inFile.read(reinterpret_cast<char *>(&image_size), sizeof(image_size));
  inFile.read(reinterpret_cast<char *>(ustar_pointer), sizeof(int) * image_size);
  inFile.close();

  inFile.open("./input/small/AERODYNAMIC_RESISTANCE.dat", std::ios::binary);
  inFile.read(reinterpret_cast<char *>(&image_size), sizeof(image_size));
  inFile.read(reinterpret_cast<char *>(aerodynamic_pointer), sizeof(int) * image_size);
  inFile.close();

  return "P2 - INITIAL RAH, 0, 0, 0\n";
}

string rah_correction_function_blocks(double ndvi_min, double ndvi_max, Candidate hot_pixel, Candidate cold_pixel, int threads_per_block)
{
  system_clock::time_point begin_core, end_core;
  int64_t general_time_core, initial_time_core, final_time_core;

  // ========= CUDA Setup
  int dev = 0;
  cudaDeviceProp deviceProp;
  HANDLE_ERROR(cudaGetDeviceProperties(&deviceProp, dev));
  HANDLE_ERROR(cudaSetDevice(dev));

  int num_threads = threads_per_block;
  int num_blocks = ceil(width_band * height_band / num_threads);

  double hot_pixel_aerodynamic = aerodynamic_pointer[hot_pixel.line * width_band + hot_pixel.col];
  hot_pixel.aerodynamic_resistance.push_back(hot_pixel_aerodynamic);

  double cold_pixel_aerodynamic = aerodynamic_pointer[cold_pixel.line * width_band + cold_pixel.col];
  cold_pixel.aerodynamic_resistance.push_back(cold_pixel_aerodynamic);

  double fc_hot = 1 - pow((ndvi_vector[hot_pixel.line][hot_pixel.col] - ndvi_max) / (ndvi_min - ndvi_max), 0.4631);
  double fc_cold = 1 - pow((ndvi_vector[cold_pixel.line][cold_pixel.col] - ndvi_max) / (ndvi_min - ndvi_max), 0.4631);

  double *devZom, *devTS, *devUstarR, *devUstarW, *devRahR, *devRahW, *devD0, *devKB1, *devH;
  HANDLE_ERROR(cudaMalloc((void **)&devZom, nBytes_band));
  HANDLE_ERROR(cudaMalloc((void **)&devD0, nBytes_band));
  HANDLE_ERROR(cudaMalloc((void **)&devKB1, nBytes_band));
  HANDLE_ERROR(cudaMalloc((void **)&devTS, nBytes_band));
  HANDLE_ERROR(cudaMalloc((void **)&devUstarR, nBytes_band));
  HANDLE_ERROR(cudaMalloc((void **)&devUstarW, nBytes_band));
  HANDLE_ERROR(cudaMalloc((void **)&devRahR, nBytes_band));
  HANDLE_ERROR(cudaMalloc((void **)&devRahW, nBytes_band));
  HANDLE_ERROR(cudaMalloc((void **)&devH, nBytes_band));

  for (int i = 0; i < 2; i++)
  {
    rah_ini_pq_terra = hot_pixel.aerodynamic_resistance[i];
    rah_ini_pf_terra = cold_pixel.aerodynamic_resistance[i];

    double LEc_terra = 0.55 * fc_hot * (hot_pixel.net_radiation - hot_pixel.soil_heat_flux) * 0.78;
    double LEc_terra_pf = 1.75 * fc_cold * (cold_pixel.net_radiation - cold_pixel.soil_heat_flux) * 0.78;

    H_pf_terra = cold_pixel.net_radiation - cold_pixel.soil_heat_flux - LEc_terra_pf;
    double dt_pf_terra = H_pf_terra * rah_ini_pf_terra / (RHO * SPECIFIC_HEAT_AIR);

    H_pq_terra = hot_pixel.net_radiation - hot_pixel.soil_heat_flux - LEc_terra;
    double dt_pq_terra = H_pq_terra * rah_ini_pq_terra / (RHO * SPECIFIC_HEAT_AIR);

    double b = (dt_pq_terra - dt_pf_terra) / (hot_pixel.temperature - cold_pixel.temperature);
    double a = dt_pf_terra - (b * (cold_pixel.temperature - 273.15));

    HANDLE_ERROR(cudaMemcpy(devTS, surface_pointer, nBytes_band, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(devZom, zom_pointer, nBytes_band, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(devD0, d0_pointer, nBytes_band, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(devKB1, kb1_pointer, nBytes_band, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(devUstarR, ustar_pointer, nBytes_band, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(devRahR, aerodynamic_pointer, nBytes_band, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(devH, sensible_heat_pointer, nBytes_band, cudaMemcpyHostToDevice));

    // ==== Paralelization core
    begin_core = system_clock::now();
    initial_time_core = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

    rah_correction_cycle_STEEP<<<num_blocks, num_threads>>>(devTS, devD0, devKB1, devZom, devUstarR, devUstarW, devRahR, devRahW, devH, a, b, height_band, width_band);
    HANDLE_ERROR(cudaDeviceSynchronize());
    HANDLE_ERROR(cudaGetLastError());

    end_core = system_clock::now();
    general_time_core = duration_cast<nanoseconds>(end_core - begin_core).count();
    final_time_core = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();
    // ====

    HANDLE_ERROR(cudaMemcpy(ustar_pointer, devUstarW, nBytes_band, cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(aerodynamic_pointer, devRahW, nBytes_band, cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(sensible_heat_pointer, devH, nBytes_band, cudaMemcpyDeviceToHost));

    double rah_hot = aerodynamic_pointer[hot_pixel.line * width_band + hot_pixel.col];
    hot_pixel.aerodynamic_resistance.push_back(rah_hot);

    double rah_cold = aerodynamic_pointer[cold_pixel.line * width_band + cold_pixel.col];
    cold_pixel.aerodynamic_resistance.push_back(rah_cold);
  }

  HANDLE_ERROR(cudaFree(devZom));
  HANDLE_ERROR(cudaFree(devD0));
  HANDLE_ERROR(cudaFree(devKB1));
  HANDLE_ERROR(cudaFree(devTS));
  HANDLE_ERROR(cudaFree(devUstarR));
  HANDLE_ERROR(cudaFree(devUstarW));
  HANDLE_ERROR(cudaFree(devRahR));
  HANDLE_ERROR(cudaFree(devRahW));
  HANDLE_ERROR(cudaDeviceReset());

  return "P2 - RAH - PARALLEL - CORE, " + to_string(general_time_core) + ", " + to_string(initial_time_core) + ", " + to_string(final_time_core) + "\n";
}

int main(int argc, char *argv[])
{
  int THREADS_PER_BLOCK = 1024;

  string result = "";
  system_clock::time_point begin, end;
  int64_t general_time, initial_time, final_time;

  begin = system_clock::now();
  initial_time = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();

  result += compute_initial_products();                                                                   // Initial Products ( Pre computed )
  result += select_endmembers(0);                                                                         // Pixel Selection  ( 0 - STEEP, 1 - ASEBAL )
  result += compute_initial_rah();                                                                        // Initial Rah      ( Pre computed )
  result += rah_correction_function_blocks(ndvi_min, ndvi_max, hot_pixel, cold_pixel, THREADS_PER_BLOCK); // Rah correction   ( RAH Cycle )

  end = system_clock::now();
  general_time = duration_cast<milliseconds>(end - begin).count();
  final_time = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();

  result += "P2 - TOTAL TIME, " + to_string(general_time) + ", " + to_string(initial_time) + ", " + to_string(final_time) + "\n";
  std::cout << result << std::endl;

  return 0;
};
