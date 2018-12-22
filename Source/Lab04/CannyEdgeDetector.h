#pragma once
#include<opencv2/core/mat.hpp>
#include<vector>
#include<math.h>
#include<iostream>
#include<utility>
using namespace std;
using namespace cv;

#define LoopImage(x,y,width,height) for(int y = 0; y < height; y++) for(int x = 0; x < width; x++)



class CannyEdgeDetector
{
	//ngưỡng dưới
	int _lowThreshold;
	//ngưỡng trên
	int _highThreshold;

	/*
	helper functions to determine if a pixel is on a strong edge, weak edge or not on an edge
	val: intensity value of pixel
	*/
	bool isStrongEdge(const int &val) { return (val > this->_highThreshold); }
	bool isWeakEdge(const int &val) { return (val > this->_lowThreshold && val < this->_highThreshold); }
	bool isNotEdge(const int &val) { return (val < this->_lowThreshold); }

	/*
	A recursive blob analysis function to preserve weak edges
	that are connected to a strong edge

	srcImg: input image
	dstImg: output image
	x, y: coordinates of a pixel on a strong edge
	*/
	void CannyEdgeDetector::rescueNeighbors(const Mat &srcImg, Mat &dstImg, int x, int y);

	/*
	Calculate a Gaussian Kernel with given dimensions and sigmas
	by precomputing separate x and y Gaussian curves

	rows: number of rows in kernel
	cols: number of columns in kernel
	sigmaX: standard deviation in X direction
	sigmaY: standard deviation in Y direction
	*/
	vector<float> getGaussianKernel(int rows, int cols, double sigmaX, double sigmaY);
public:
	/*
	Hàm áp dụng thuật toán Canny để phát hiện biên cạnh
	- srcImage: ảnh input
	- dstImage: ảnh kết quả
	Hàm trả về
	1: nếu detect thành công
	0: nếu detect không thành công
	*/
	int Apply(const Mat& srcImage, Mat &dstImage);

	/*
	Apply a Gaussian filter to pre-smooth image

	srcImage: input image
	dstImg: output image
	*/
	void preSmooth(const Mat &srcImage, Mat &dstImage);

	/*
	Compute the intensity gradient of image using Sobel operator
	
	srcImage: input image
	dstImg: output image
	*/
	void computeGradient(const Mat& srcImage, Mat &magnitude, Mat &theta);

	/*
	Suppress non-maximum pixels to make thinner edges

	magnitude: gradient magnitude
	theta: representation of gradient direction
	*/
	int CannyEdgeDetector::NonMaxSuppress(const Mat &magnitude, const Mat &theta, Mat &dstImage);

	/*
	Apply double-thresholding to eliminate weak edges
	
	srcImage: input image
	dstImg: output image
	*/
	void Hysteresis(const Mat &srcImage, Mat &dstImage);

	void setThresholds(int low, int high);

	CannyEdgeDetector();
	~CannyEdgeDetector();
};

