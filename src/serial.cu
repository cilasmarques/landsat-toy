#include <iostream>
#include <chrono>
#include <vector>
#include <thread>
#include <tiffio.h>
#include <string>

using namespace std::chrono;

// Define the matrix size
#define MATRIX_HEIGTH 128 * 64
#define MATRIX_WIDTH 128 * 64

int main()
{
  int64_t general_time;
  TIFF *bands_resampled[8];
  system_clock::time_point begin, end;

  // TIFFs Setup
  std::string bands_paths[] = {
      "",
      "./input/scenes/B2.tif",
      "./input/scenes/B3.tif",
      "./input/scenes/B4.tif",
      "./input/scenes/B5.tif",
      "./input/scenes/B6.tif",
      "./input/scenes/B10.tif",
      "./input/scenes/B7.tif",
      "./input/scenes/final_tal.tif"};

  for (int i = 1; i <= 8; i++)
  {
    std::string path_tiff_base = bands_paths[i];
    bands_resampled[i] = TIFFOpen(path_tiff_base.c_str(), "rm");
  }

  uint16_t sample_format;
  uint32_t height, width;
  TIFFGetField(bands_resampled[1], TIFFTAG_IMAGELENGTH, &height);
  TIFFGetField(bands_resampled[1], TIFFTAG_IMAGEWIDTH, &width);
  TIFFGetField(bands_resampled[1], TIFFTAG_SAMPLEFORMAT, &sample_format);

  // Read TIFFs
  double *band1 = new double[MATRIX_HEIGTH * MATRIX_WIDTH];
  double *band2 = new double[MATRIX_HEIGTH * MATRIX_WIDTH];
  double *band3 = new double[MATRIX_HEIGTH * MATRIX_WIDTH];
  double *band4 = new double[MATRIX_HEIGTH * MATRIX_WIDTH];
  double *band5 = new double[MATRIX_HEIGTH * MATRIX_WIDTH];
  double *band6 = new double[MATRIX_HEIGTH * MATRIX_WIDTH];
  double *band7 = new double[MATRIX_HEIGTH * MATRIX_WIDTH];
  double *band8 = new double[MATRIX_HEIGTH * MATRIX_WIDTH];

  std::vector<double *> bands;
  bands.push_back(band1);
  bands.push_back(band2);
  bands.push_back(band3);
  bands.push_back(band4);
  bands.push_back(band5);
  bands.push_back(band6);
  bands.push_back(band7);
  bands.push_back(band8);

  for (int i = 1; i <= 8; i++)
  {
    for (int line = 0; line < height; line++)
    {
      TIFF *curr_band = bands_resampled[i];
      tdata_t band_line_buff = _TIFFmalloc(TIFFScanlineSize(curr_band));
      unsigned short curr_band_line_size = TIFFScanlineSize(curr_band) / width;
      TIFFReadScanline(curr_band, band_line_buff, line);

      for (int col = 0; col < width; col++)
      {
        float value = 0;
        memcpy(&value, static_cast<unsigned char *>(band_line_buff) + col * curr_band_line_size, curr_band_line_size);
        bands[i - 1][line * width + col] = value;
      }
      _TIFFfree(band_line_buff);
    }
  }

  // ======== RUN =========

  begin = system_clock::now();

  for (int i = 0; i < 1; i++)
  {
    for (int j = 0; j < MATRIX_WIDTH; j++)
    {
      std::cout << bands[i][j] << " ";
    }
    std::cout << std::endl;
  }

  end = system_clock::now();
  general_time = duration_cast<nanoseconds>(end.time_since_epoch() - begin.time_since_epoch()).count();
  std::cout << "CUDA CORE," << MATRIX_HEIGTH << " x " << MATRIX_WIDTH << ", " << general_time << std::endl;
  // ======== RUN =========

  return 0;
}