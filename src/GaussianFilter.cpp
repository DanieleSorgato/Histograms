#include "opencv2/highgui/highgui.hpp"
#include "Filter.hpp"
#include <iostream>

GaussianFilter::GaussianFilter(cv::Mat input_img, int filter_size):Filter(input_img, filter_size)
{
	//initialize sigma to 0
	sigma = 0;
}
void GaussianFilter::doFilter()
{
	std::cout << "apply gaussian filtering" << std::endl;
	cv::GaussianBlur(input_image, result_image, cv::Size(filter_size, filter_size), sigma, sigma);
}
void GaussianFilter::setSigma(double s)
{
	sigma = s;
}
double GaussianFilter::getSigma()
{
	return sigma;
}