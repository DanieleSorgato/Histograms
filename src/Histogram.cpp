#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include "Filter.hpp"

using namespace std;
using namespace cv;

//FIRST PART OF THE EXERCISE

void showHistogram(std::vector<cv::Mat>& hists)
{
	// Min/Max computation
	double hmax[3] = { 0,0,0 };
	double min;
	cv::minMaxLoc(hists[0], &min, &hmax[0]);
	cv::minMaxLoc(hists[1], &min, &hmax[1]);
	cv::minMaxLoc(hists[2], &min, &hmax[2]);

	std::string wname[3] = { "blue", "green", "red" };
	cv::Scalar colors[3] = { cv::Scalar(255,0,0), cv::Scalar(0,255,0),
							 cv::Scalar(0,0,255) };

	std::vector<cv::Mat> canvas(hists.size());

	// Display each histogram in a canvas
	for (int i = 0, end = hists.size(); i < end; i++)
	{
		canvas[i] = cv::Mat::ones(125, hists[0].rows, CV_8UC3);

		for (int j = 0, rows = canvas[i].rows; j < hists[0].rows - 1; j++)
		{
			cv::line(
				canvas[i],
				cv::Point(j, rows),
				cv::Point(j, rows - (hists[i].at<float>(j) * rows / hmax[i])),
				hists.size() == 1 ? cv::Scalar(200, 200, 200) : colors[i],
				1, 8, 0
			);
		}

		cv::imshow(hists.size() == 1 ? "value" : wname[i], canvas[i]);
	}
}

//function to calculate histograms
void elabHisto(vector<Mat>& bgr)
{
	int histSize = 256;
	float range[] = { 0, 255 };
	const float* histRange = { range };
	bool uniform = true;
	bool accumulate = false;

	Mat b_hist, g_hist, r_hist;
	calcHist(&bgr[0], 1, 0, Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate);
	calcHist(&bgr[1], 1, 0, Mat(), g_hist, 1, &histSize, &histRange, uniform, accumulate);
	calcHist(&bgr[2], 1, 0, Mat(), r_hist, 1, &histSize, &histRange, uniform, accumulate);
	vector<cv::Mat> hists = { b_hist, g_hist, r_hist };
	showHistogram(hists);
}

//funzione per elaborazione in spazio hsv
void hsvEqualization(Mat image, vector<Mat> hsv, int component)
{
	Mat HSV_final;
	vector<Mat> histograms;
	split(image, hsv);

	//equalizzo il canale richiesto
	equalizeHist(hsv[component], hsv[component]); //modifico solo la componente richiesta
	merge(hsv, HSV_final);

	//calcolo istogrammi
	elabHisto(hsv);
	cvtColor(HSV_final, HSV_final, COLOR_HSV2BGR, 0);
	imshow("hsv", HSV_final);
}




//SECOND PART OF THE EXERCISE


//struttura per i dati, i 2 parametri assumono significati diversi a seconda della classe
struct ImageWithParams
{
	int param1;
	int param2;
	cv::Mat src_img, filtered_img;
};
ImageWithParams dati;

//functions to manipulate trackbar events
void gaussKernel(int x, void* a)
{
	//in this function I modify the first parameter of the structure and I use the second like it is
	ImageWithParams str = *(static_cast<ImageWithParams*>(a));
	dati = str;
	dati.param1 = x;

	GaussianFilter gauss = GaussianFilter(dati.src_img, dati.param1);
	gauss.setSigma(dati.param2);
	gauss.doFilter();
	imshow("gaussian", gauss.getResult());
}
void gaussSigma(int x, void* a)
{
	//in this function I modify the second parameter of the structure and I use the first like it is
	ImageWithParams str = *(static_cast<ImageWithParams*>(a));
	dati = str;
	dati.param2 = x;

	GaussianFilter gauss = GaussianFilter(dati.src_img, dati.param1);
	gauss.setSigma(dati.param2);
	gauss.doFilter();
	imshow("gaussian", gauss.getResult());
}
void trackMed(int x, void* a)
{
	//for median filter I need only one function because I use only one parameter
	ImageWithParams str = *(static_cast<ImageWithParams*>(a));
	dati = str;
	dati.param1 = x;

	MedianFilter med = MedianFilter(dati.src_img, dati.param1);
	med.doFilter();
	imshow("median", med.getResult());
}
void bilatSigmaR(int x, void* a)
{
	//here I modify first parameter
	ImageWithParams str = *(static_cast<ImageWithParams*>(a));
	dati = str;
	dati.param1 = x;

	BilateralFilter bilat = BilateralFilter(dati.src_img, 25);
	bilat.setSigmaR(dati.param1);
	bilat.setSigmaS(dati.param2);
	bilat.doFilter();
	imshow("bilateral", bilat.getResult());
}
void bilatSigmaS(int x, void* a)
{
	//here I modify second parameter
	ImageWithParams str = *(static_cast<ImageWithParams*>(a));
	dati = str;
	dati.param2 = x;

	BilateralFilter bilat = BilateralFilter(dati.src_img, 25);
	bilat.setSigmaR(dati.param1);
	bilat.setSigmaS(dati.param2);
	bilat.doFilter();
	imshow("bilateral", bilat.getResult());
}


 




int main()
{
	Mat img = imread("../data/image.jpg");

	//split image in three images of one color: B, G and R
	vector<Mat> bgr;
	split(img, bgr);

	//elaborate histograms and visualize original image
	elabHisto(bgr);
	resize(img, img, Size(img.cols / 3.0, img.rows / 3.0));
	imshow("original", img);

	waitKey(0);
	destroyWindow("original");



	//equalize the image for each colour
	vector<Mat> equal = bgr; //I initialize it because it needs to be the same dimension of bgr
	Mat final_img;
	
	equalizeHist(bgr[0], equal[0]);
	equalizeHist(bgr[1], equal[1]);
	equalizeHist(bgr[2], equal[2]);

	//recreate equalized image and elaborate histograms
	merge(equal, final_img);
	elabHisto(equal);
	resize(final_img, final_img, Size(final_img.cols / 3.0, final_img.rows / 3.0));
	imshow("equalized image", final_img);

	waitKey(0);
	destroyWindow("equalized image");





	// execute test in HSV space
	Mat HSV_img;
	vector<Mat> hsv;
	cvtColor(img, HSV_img, COLOR_BGR2HSV, 0);

	//equalize one by one channel
	split(HSV_img, hsv);
	cout << "image equalizedin channel H" << endl;
	hsvEqualization(HSV_img, hsv, 0);
	waitKey(0);

	cout << "image equalizedin channel S" << endl;
	hsvEqualization(HSV_img, hsv, 1);
	waitKey(0);

	cout << "image equalizedin channel V" << endl;
	hsvEqualization(HSV_img, hsv, 2);
	waitKey(0);



	destroyAllWindows();  //RESET
	waitKey(500);


	Mat nuova = imread("../data/lena.png");
	//create a copy of the image for each filter to be applied
	Mat median = nuova.clone();
	Mat gaussian = nuova.clone();
	Mat bilateral = nuova.clone();
	imshow("original", nuova);

	//values for trackbars
	int value = 0;
	int maxKernel = 10;
	int maxSigma = 80;

	dati.param1 = 0;
	dati.param2 = 0;

	dati.src_img = gaussian;
	GaussianFilter myGauss = GaussianFilter(gaussian, 1); //it needs only to show the first image in spite of an empty window
	myGauss.doFilter();
	namedWindow("gaussian", WINDOW_AUTOSIZE);
	imshow("gaussian", myGauss.getResult());
	createTrackbar("gaussian_Kernel", "gaussian", &value, maxKernel, gaussKernel, &dati);
	createTrackbar("gaussian_sigma", "gaussian", &value, maxSigma, gaussSigma, &dati);

	dati.src_img = median;
	MedianFilter myMed = MedianFilter(median, 5);
	myMed.doFilter();
	namedWindow("median", WINDOW_AUTOSIZE);
	imshow("median", myMed.getResult());
	createTrackbar("median_Kernel", "median", &value, maxKernel, trackMed, &dati);

	dati.src_img = bilateral;
	BilateralFilter myBil = BilateralFilter(bilateral, 25);
	myBil.doFilter();
	namedWindow("bilateral", WINDOW_AUTOSIZE);
	imshow("bilateral", myBil.getResult());
	createTrackbar("bilateral_sigmaR", "bilateral", &value, maxSigma, bilatSigmaR, &dati);
	createTrackbar("bilateral_sigmaS", "bilateral", &value, maxSigma, bilatSigmaS, &dati);
	
	waitKey(0);
}