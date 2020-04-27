#include "opencv2/highgui/highgui.hpp"
#include "Filter.hpp"
#include <iostream>

BilateralFilter::BilateralFilter(cv::Mat input_img, int filter_size) : Filter(input_img, filter_size)
{
	//set sigma values to 0
	sigma_r = 0;
	sigma_s = 0;
}
void BilateralFilter::doFilter()
{
	std::cout << "apply bilateral filtering" << std::endl;
	cv::bilateralFilter(input_image, result_image, filter_size, sigma_r, sigma_s);
}
void BilateralFilter::setSigmaR(double r)
{
	sigma_r = r;
}
void BilateralFilter::setSigmaS(double s)
{
	sigma_s = s;
}