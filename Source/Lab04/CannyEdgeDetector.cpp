#include "CannyEdgeDetector.h"

void CannyEdgeDetector::setThresholds(int low, int high)
{
    this->_lowThreshold = low;
    this->_highThreshold = high;
}

void CannyEdgeDetector::rescueNeighbors(const Mat &srcImg, Mat &dstImg, int x, int y)
{
    // Find neighboring weak edges
    vector< pair<int, int> > neighborEdges;

    for (int _it_x = -1; _it_x <= 1; ++_it_x)
    {
        for (int _it_y = -1; _it_y <= 1; ++_it_y)
        {
            if (_it_x == 0 && _it_y == 0) // center pixel
                continue;

            if (isWeakEdge(dstImg.at<uchar>(y + _it_y, x + _it_x)))
                neighborEdges.push_back(make_pair(x + _it_x, y + _it_y));
        }
    }


    // If there is no neighboring weak edges
    if (neighborEdges.size() == 0)
        return; // then terminate

    // Else preserve edges and recurse
    for (int i = 0; i < neighborEdges.size(); ++i)
    {
        int newX = get<0>(neighborEdges[i]);
        int newY = get<1>(neighborEdges[i]);
        // Preserve edges
        dstImg.at<uchar>(newY, newX) = 255;

        // Recurse to neighbor pixels        
        rescueNeighbors(srcImg, dstImg, newX, newY);
    }
}

vector<float> CannyEdgeDetector::getGaussianKernel(int rows, int cols, double sigmaX, double sigmaY)
{
	// Calculate 2 separate Gaussian curves
	// (Lots of math, don't even bother)

    const auto y_mid = (rows - 1) / 2.0;
    const auto x_mid = (cols - 1) / 2.0;

    const auto x_spread = 1. / (sigmaX * sigmaX * 2);
    const auto y_spread = 1. / (sigmaY * sigmaY * 2);

    const auto denominator = 8 * atan(1) * sigmaX * sigmaY;

    vector<float> gauss_x, gauss_y, gauss;

    gauss_x.reserve(cols);
    for (auto i = 0; i < cols; ++i)
    {
        auto x = i - x_mid;
        gauss_x.push_back(exp(-x * x * x_spread));
    }

    gauss_y.reserve(rows);
    for (auto i = 0; i < rows; ++i)
    {
        auto y = i - y_mid;
        gauss_y.push_back(exp(-y * y * y_spread));
    }

    // Compute Gaussian kernel from Gaussian curves

    gauss.reserve(rows * cols);
    for (int j = 0; j < rows; ++j)
    {
        for (int i = 0; i < cols; ++i)
        {
            gauss.push_back(gauss_x[i] * gauss_y[j] / denominator);
        }
    }

    return gauss;
}

void CannyEdgeDetector::preSmooth(const Mat &srcImage, Mat &dstImage)
{
    dstImage = srcImage.clone();

    vector<float> GaussianKernel = this->getGaussianKernel(5, 5, 1, 1);
    int xOff = 2, yOff = 2;
	LoopImage(x, y, srcImage.cols, srcImage.rows)
	{
        float sum = 0.f;
        for (int k = -2; k <= 2; ++k)
        {
            for (int l = -2; l <= 2; ++l)
            {
				// Ignore edges
				if ((x + k < 0) || (x + k > srcImage.cols - 1) || (y + l < 0) || (y + l > srcImage.rows - 1))
					continue;

				int kernelOffset = (k + xOff) * 5 + (l + yOff);
                sum += srcImage.at<uchar>(l + y, k + x) * GaussianKernel[kernelOffset];
            }
        }
        dstImage.at<uchar>(y, x) = (int)sum;
	}
}

void CannyEdgeDetector::computeGradient(const Mat& srcImage, Mat &magnitude, Mat &theta)
{
    // Preparations
    Mat xGradient = srcImage.clone(), yGradient = srcImage.clone();
    magnitude = srcImage.clone();
    theta = Mat(srcImage.rows, srcImage.cols, CV_32F);
    for (int y = 0; y < srcImage.rows; ++y)
    {
        for (int x = 0; x < srcImage.cols; ++x)
        {
            xGradient.at<uchar>(y,x) = 0;
            yGradient.at<uchar>(y,x) = 0;
            magnitude.at<uchar>(y,x) = 0;
            theta.at<float>(y,x) = 0;
        }
    }

    vector<float> kerX =
    { 
        1.0 ,  0,  -1.0 ,
        2.0 ,  0,  -2.0 ,
        1.0 ,  0,  -1.0  
    };

    vector<float> kerY = 
    {
        -1.0,   -2.0,   -1.0,
        0   ,      0,      0, 
        1.0 ,    2.0,    1.0 
    };

    // Loop through pixels in image
    int xOff = 1, yOff = 1;
    LoopImage(x, y, srcImage.cols, srcImage.rows)
	{
		// Ignore edges
		if ((x - xOff < 0) || (x + xOff > srcImage.cols - 1) ||
			(y - yOff < 0) || (y + yOff > srcImage.rows - 1))
			continue;

        float gx = 0, gy = 0, g = 0, _theta = 0;

        for (int k = -1; k <= 1; ++k)
        {
            for (int l = -1; l <= 1; ++l)            
			{
				int kernelOffset = (k + xOff) * 3 + (l + yOff);
                gx += srcImage.at<uchar>(l + y, k + x) * kerX[kernelOffset]; // x gradient
                gy += srcImage.at<uchar>(l + y, k + x) * kerY[kernelOffset]; // y gradient
            }
        }

        g = hypot(gx, gy);
        if (gx == 0 && gy == 0)
            _theta = 0.f;
        else _theta = atan2(gy, gx);

        magnitude.at<uchar>(y, x) = (int)g;
        theta.at<float>(y, x) = _theta;
    }
}

int CannyEdgeDetector::NonMaxSuppress(const Mat &magnitude, const Mat &theta, Mat &dstImage)
{
    dstImage = magnitude.clone();

    const double _PI = 3.14159265358979323846;
    int xOff = 1, yOff = 1;
    LoopImage(x, y, magnitude.cols, magnitude.rows)
	{
        // Ignore edges
        if ((x - xOff < 0) || (x + xOff > magnitude.cols - 1) ||
            (y - yOff < 0) || (y + yOff > magnitude.rows - 1))
			continue;

        // Convert theta from radians to degrees
        float angle = theta.at<float>(y, x) * 180 / _PI;
        int iAngle = 0;
        // Approximate angle value to intervals of 45 degrees
        if (((angle<22.5) && (angle>-22.5)) || (angle > 157.5) || (angle < -157.5)) {
            iAngle = 0;
        }
        else if (((angle > 22.5) && (angle < 67.5)) || (angle < -112.5) || (angle > -157.5)) {
            iAngle = 45;
        }
        else if (((angle > 67.5) && (angle < 112.5)) || (angle < -67.5) || (angle > -112.5)) {
            iAngle = 90;
        }
        else if (((angle > 112.5) && (angle < 157.5)) || (angle < -22.5) || (angle > -67.5)) {
            iAngle = 135;
        }

        int neighbor1, neighbor2;
        switch (iAngle)
        {
            case 0:
            {
                neighbor1 = magnitude.at<uchar>(y, x-1);
                neighbor2 = magnitude.at<uchar>(y, x+1);
            }
            break;
            case 45:
            {
                neighbor1 = magnitude.at<uchar>(y-1, x-1);
                neighbor2 = magnitude.at<uchar>(y+1, x+1);
            }
            break;
            case 90:
            {
                neighbor1 = magnitude.at<uchar>(y-1, x);
                neighbor2 = magnitude.at<uchar>(y+1, x);
            }
            break;
            case 135:
            {
                neighbor1 = magnitude.at<uchar>(y+1, x-1);
                neighbor2 = magnitude.at<uchar>(y-1, x+1);
            }
            break;
            default:
            {
                cout << "Error: invalid gradient direction" << endl;
                return 0;
            }
        }

        // Preserve local maxima and suppress non-maxima
        int thisPixel = magnitude.at<uchar>(y, x);
        if (thisPixel < neighbor1 || thisPixel < neighbor2) // if this pixel is not local maximum
        {
            dstImage.at<uchar>(y, x) = 0; // then suppress it
        }
        // else don't suppress it (do nothing)        
    }

	return 1;
}

void CannyEdgeDetector::Hysteresis(const Mat &srcImage, Mat &dstImage)
{
    dstImage = srcImage.clone();

    int xOff = 1, yOff = 1;
    // First loop (Detect strong edges, preserve connecting weak edges, eliminate non-edges)
    LoopImage(x, y, srcImage.cols, srcImage.rows)
	{
        // Ignore edges
        if ((x - xOff < 0) || (x + xOff > srcImage.cols - 1) ||
            (y - yOff < 0) || (y + yOff > srcImage.rows - 1))
			continue;

		int thisPixel = dstImage.at<uchar>(y, x);

        // Preserve strong edges
        if (this->isStrongEdge(thisPixel))
        {
            dstImage.at<uchar>(y, x) = 255;
            this->rescueNeighbors(srcImage, dstImage, x, y);
        }
        // Suppress non-edges
        else if (this->isNotEdge(thisPixel))
        {
            dstImage.at<uchar>(y, x) = 0;
        }
    }

    // Second loop (eliminate false-positive weak edges)
    LoopImage(x, y, srcImage.cols, srcImage.rows)
    {
		int thisPixel = dstImage.at<uchar>(y, x);

        if (!(this->isStrongEdge(thisPixel)))
        {
            dstImage.at<uchar>(y, x) = 0;
        }
    }
}

int CannyEdgeDetector::Apply(const Mat& srcImage, Mat &dstImage)
{
	Mat smoothed, magnitude, theta, suppressed;

	// Pre-smooth image
    cout << "Applying Gaussian filter..." << endl;
	this->preSmooth(srcImage, smoothed);

    // Compute intensity gradient
    cout << "Computing intensity gradient..." << endl;
    this->computeGradient(smoothed, magnitude, theta);

    // Non-maximum suppression
    cout << "Suppressing non-maximum edges..." << endl;
    int ret = 
    this->NonMaxSuppress(magnitude, theta, suppressed);

    // Hysteresis double-thresholding
    cout << "Applying hysteresis double-thresholding..." << endl;
    if (ret == 1) // If non-max suppression was successfully performed
    this->Hysteresis(suppressed, dstImage);

	return ret;
}

CannyEdgeDetector::CannyEdgeDetector()
{
}


CannyEdgeDetector::~CannyEdgeDetector()
{
}
