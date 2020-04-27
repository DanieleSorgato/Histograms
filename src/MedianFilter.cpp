#include "opencv2/highgui/highgui.hpp"
#include "Filter.hpp"
#include <iostream>

MedianFilter::MedianFilter(cv::Mat input_img, int filter_size) : Filter(input_img, filter_size)
{
	//no constructor is needed
}
void MedianFilter::doFilter()
{

	std::cout << "apply median filtering" << std::endl;
	cv::medianBlur(input_image, result_image, filter_size);
}