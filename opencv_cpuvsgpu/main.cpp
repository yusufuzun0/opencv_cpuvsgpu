
#include <iostream>
#include <chrono>

#include <opencv2/core/core.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudafilters.hpp>

#define WINDOW_SIZE 25
void medianfilter_cpu_opencv(const std::string& filename);
void medianfilter_gpu_opencv(const std::string& filename);

int main()
{
	const std::string input_filename = "C:/Dechard/Görüntüler/lenabozuk.png";

   

	auto start_time_cpu_opencv = std::chrono::high_resolution_clock::now();
	medianfilter_cpu_opencv(input_filename);
	auto end_time_cpu_opencv = std::chrono::high_resolution_clock::now();

	auto start_time_gpu_opencv = std::chrono::high_resolution_clock::now();
	medianfilter_gpu_opencv(input_filename);
	auto end_time_gpu_opencv = std::chrono::high_resolution_clock::now();

	std::cout << "OpenCV-CPU Execution Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_time_cpu_opencv - start_time_cpu_opencv).count() << " ms" << std::endl;
	std::cout << "OpenCV-GPU Execution Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_time_gpu_opencv - start_time_gpu_opencv).count() << " ms" << std::endl;

    return EXIT_SUCCESS;
}

void medianfilter_cpu_opencv(const std::string& filename)
{
	cv::Mat image = cv::imread(filename);
	cv::Mat filtered_image;
	cv::medianBlur(image, filtered_image, WINDOW_SIZE);

	cv::imwrite("C:/Dechard/opencv_cpu_median.jpg", filtered_image);

}

void medianfilter_gpu_opencv(const std::string& filename)
{
	cv::Mat input_image = cv::imread(filename, cv::IMREAD_COLOR);
	if (input_image.channels() == 3)
	{
		std::cerr << "Input image must have 3 channels." << std::endl;
		return;
	}
	cv::cuda::GpuMat gpu_input_image;

	gpu_input_image.upload(input_image); //Upload image to GPU memory

	cv::cuda::GpuMat gpu_median_filter_result;
	cv::Ptr<cv::cuda::Filter> median_filter = cv::cuda::createMedianFilter(cv::IMREAD_COLOR, WINDOW_SIZE);
	median_filter->apply(gpu_input_image, gpu_median_filter_result);
	cv::Mat output_image;

	gpu_median_filter_result.download(output_image); //Download result to CPU memory

	cv::imwrite("C:/Dechard/opencv_gpu_median.jpg", output_image);

}