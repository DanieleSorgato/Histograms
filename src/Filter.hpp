#pragma once
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

// Generic class implementing a filter with the input and output image data and the parameters
class Filter {

	// Methods

public:

	// constructor 
	// input_img: image to be filtered
	// filter_size : size of the kernel/window of the filter
	Filter(cv::Mat input_img, int filter_size);

	// perform filtering (in base class do nothing, to be reimplemented in the derived filters)
	void doFilter();

	// get the output of the filter
	cv::Mat getResult();

	//set the window size (square window of dimensions size x size)
	void setSize(int size);

	//get the Window Size
	int getSize();

	// Data

protected:

	// input image
	cv::Mat input_image;

	// output image (filter result)
	cv::Mat result_image;

	// window size
	int filter_size;



};

// Gaussian Filter
class GaussianFilter : public Filter {

	// write here your code

	// place constructor
	// re-implement  doFilter()
	// additional parameter: standard deviation (sigma)
public:
	GaussianFilter(cv::Mat, int);
	void doFilter();
	void setSigma(double s);
	double getSigma();
	double sigma;

};

class MedianFilter : public Filter {

	// write here your code

	// place constructor
	// re-implement  doFilter()
	// no additional parameters
public:
	MedianFilter(cv::Mat, int);
	void doFilter();
};

class BilateralFilter : public Filter {

	// write here your code

	// place constructor
	// re-implement  doFilter()
	// additional parameters: sigma_range, sigma_space
public:
	BilateralFilter(cv::Mat, int);
	void doFilter();
	void setSigmaR(double);
	void setSigmaS(double);
	double sigma_r;
	double sigma_s;

};