#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include "CannyEdgeDetector.h"
#include "Lexer.h"
using namespace std;
using namespace cv;

int main(int argc, char *argv[]) {
	// Command-line argument structure:
	// <Program.exe> <Command> <InputPath> <CmdArguments>

	// If there are insufficient arguments
	if (argc < 5)
	{
		cout << "ERROR: Insufficient arguments." << endl;
		return -1;
	}

	// Read image from InputPath
	Mat srcImg;
	srcImg = imread(argv[2], IMREAD_GRAYSCALE); // Read the file
	if (srcImg.empty())							// Check for invalid input
	{
		cout << "Could not open or find the image" << std::endl;
		return -2;
	}

	// Read Command and CmdArguments
	// and perform tasks accordingly
	Mat dstImg;
	int result;
	Lexer *myLexer = new Lexer();
	string outputSuffix;
	switch (myLexer->lex(argv[1]))
	{
	case Canny:
	{
		int low = atoi(argv[3]);
		int high = atoi(argv[4]);
		CannyEdgeDetector *canny = new CannyEdgeDetector;
		canny->setThresholds(low, high);
		result = canny->Apply(srcImg, dstImg);
		delete canny;
		if (result == 0) return -3;
		break;
	}
	case Bad_command:
		cout << "Command not found." << endl;
		return -4;
	default:
		break;
	}
	
	cout << "Done." << endl;

	// Show results
	namedWindow("Source image", WINDOW_AUTOSIZE); // Create a window for display.	
	imshow("Source image", srcImg);					// Show source image.
	namedWindow("Destination image", WINDOW_AUTOSIZE); // Create a window for display.
	imshow("Destination image", dstImg);				// Show result image.
	waitKey(0);										// Wait for a keystroke in the window
	return 0;
}