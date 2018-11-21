#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>
#include <qututil/ImageTools.h>
#include <vul/vul_arg.h>
#include <vcl_string.h>
#include <vcl_sstream.h>
#include <vcl_iostream.h>
#include <vcl_fstream.h>
#include <qutcrowd/count/qutcrowd_count.h>
#include <qutopencv/qut_video_read.h>
#include <qutopencv/qut_video_write.h>
#include <vil/algo/vil_orientations.h>
#include <qutcfg/XMLFileDataSource.h>
#include <vil/vil_crop.h>
#include <vil/vil_save.h>
#include <qutapps/qutOpAnalytics/CrowdUtils.h>

#define HISTOGRAM_BINS 4

using namespace cv;

// ****************************************************************************
// ****************************************************************************
//  Authour: Wafi Hossain
//  This function's purpose is to train and create models for kiosk dwell time
//	by means of support vector machines. It is meant to be treated as a prototype
// ****************************************************************************
// ****************************************************************************

/*
*	This function iterates through the region of interest and
* and sums the pixel densities for the respective RegressionSetup
* hence they are arranged in a vector and returned
*/
vcl_vector<float> ExtractFeaturePerPixelCrowd(int x, int y, int height, int width, vil_image_view<double> perPixelCrowd, int numHDivs, int numVDivs, vil_image_view<unsigned char> im);
//Returns whether or not a number is contained within the list of events
bool FrameIsOccupied(vcl_vector<vcl_pair<int, int> > events, int frame);

/*
*	will hold data such as coordinates and ROI
*	for a single kiosk
*/
struct KioskData {
  int ROI[4];
	int id;
	vcl_vector<vcl_pair<int, int> > events;
	vcl_vector<vcl_pair<int, int> > testEvents;
};

/*
*	A few global variables that will be used to save the config data in
*/
int kiosks;
int frames;
int hDiv;
int wDiv;
int featureSize = 0;

/*
*	Metrics
*/
float falsePositives = 0;
float truePositives = 0;
float falseNegatives = 0;
float trueNegatives = 0;



int main(int argc, char** argv) {
	//
	// check for inputs
	//
	vul_arg<vcl_string>
		arg_crowdCounterConfig("-CCcfg", "Config File", "/home/webmaster/saivt-vxl/qut/qutapps/qutSECAnalytics/models/QAL/9996.xml"),
		arg_config("-cfg", "Config File", "/home/webmaster/saivt-vxl/qut/qutapps/qutSECAnalytics/configs/kioskTrainer.xml");
	vul_arg<int> arg_test("-test", "testing?", 0);
	vul_arg_parse(argc, argv);
	if(arg_config() == "") {
		vul_arg_display_usage_and_exit();
	}
	//
	// load config file and important data
	//
	XMLFileDataSource* config = new XMLFileDataSource(arg_config().c_str());
	config->load();
	frames = config->getInteger("/KioskData/frames-used/@frames");
	hDiv = config->getInteger("/KioskData/Divisions/@Hdiv");
	wDiv = config->getInteger("/KioskData/Divisions/@Wdiv");
	std::cout << "height divisions " << hDiv << " width divisions: " << wDiv << std::endl;
	kiosks = config->getInteger("count(/KioskData/Kiosk)");
	std::cout << "Number of kiosks: " << kiosks << std::endl;
	//
	// load the kiosks data in a vector
	//
	vcl_vector<KioskData> kd;
	for(int i = 0; i < kiosks; i++) {
		KioskData kiosk;
		kiosk.id = i+1;
		vcl_stringstream currentKiosk;
		currentKiosk << "/KioskData/Kiosk[" << (i + 1) << "]";
		kiosk.ROI[0] = config->getInteger(currentKiosk.str() + "/ROI/point/@x");
		kiosk.ROI[1] = config->getInteger(currentKiosk.str() + "/ROI/point/@y");
		kiosk.ROI[2] = config->getInteger(currentKiosk.str() + "/ROI/dimensions/@height");
		kiosk.ROI[3] = config->getInteger(currentKiosk.str() + "/ROI/dimensions/@width");
		int numEvents = config->getInteger("count(" + currentKiosk.str() +  "/Event)");
		std::cout << "Kiosk: " << i+1 << " x: " << kiosk.ROI[0] << std::endl;
		std::cout << "y: " << kiosk.ROI[1] << " height: " << kiosk.ROI[2] << " width: " << kiosk.ROI[3] << std::endl;
		//
		// load events for training
		//
		for (int c = 0; c < numEvents; c++) {
			vcl_pair<int, int> pair;
			vcl_stringstream currentEvent;
			currentEvent << currentKiosk.str() << "/Event[" << c + 1 << "]";
			pair = vcl_make_pair(config->getInteger(currentEvent.str() + "/@start"), config->getInteger(currentEvent.str() + "/@end"));
			kiosk.events.push_back(pair);
		}
		//
		// load the events for testing
		//
		numEvents = config->getInteger("count(" + currentKiosk.str() +  "/TestEvent)");
		for (int c = 0; c < numEvents; c++) {
			vcl_pair<int, int> pair;
			vcl_stringstream currentEvent;
			currentEvent << currentKiosk.str() << "/TestEvent[" << c + 1 << "]";
			pair = vcl_make_pair(config->getInteger(currentEvent.str() + "/@start"), config->getInteger(currentEvent.str() + "/@end"));
			kiosk.testEvents.push_back(pair);
		}
		kd.push_back(kiosk);
	}
	//
	// load crowd counter
	//
	XMLFileDataSource* crowdCounterConfig = new XMLFileDataSource(arg_crowdCounterConfig().c_str());
	crowdCounterConfig->load();
	qutcrowd_count* cc;
	cc = new qutcrowd_count();
	cc->TrainExtractedFeatures(crowdCounterConfig);
	std::cout << "Crowd Counter is now loaded" << std::endl;
	//
	// Save this variable to see what mode this is to be ran
	//
	int testing = arg_test();
	//
	// execute the training routine with the sample data
	//
	if(testing == 0) {
		std::cout << "Will try to create feature arrays" << std::endl;
		//
		// prepare arrays for SVM
		//
		float* labels = new float[kiosks*frames];
		float** features = new float*[kiosks*frames];
		std::cout << "feature arrays have been created" << std::endl;
		//
		// load videos
		//
		qut_video_read<unsigned char>* normalVid =  new qut_video_read<unsigned char>(config, "/KioskData/qut-video-normal");
		qut_video_read<unsigned char>* motionVid =  new qut_video_read<unsigned char>(config, "/KioskData/qut-video-motion");
		int countLabel = 0;
		//
		// create vector to hold features that will be then converted to
		// array for openCV integration
		//
		vcl_vector<vcl_vector<float> > featuresVector;
		std::cout << "Starting to extract feature" << std::endl;
		//
		// go through frames and extract features
		//
		while((normalVid->Next() && (normalVid->Frame() <= frames)) && (motionVid->Next() && (motionVid->Frame() <= frames))) {
			std::cout << "Processing Frame: " << (normalVid->Frame()+1) << std::endl;
			vil_image_view<unsigned char> im = *normalVid->Current();
			vil_image_view<unsigned char> motion = *motionVid->Current();
			cc->Process(im, &motion);
			vil_image_view<unsigned int> blobID = cc->GetBlobIDMap();
			vil_image_view<float> densityMap = cc->GetDensityMap();
			vil_image_view<double> perPixelCrowd;
			PerPixelCrowdDensity(blobID, densityMap, cc->GetBlobEstimates()[0], perPixelCrowd);
			for(int k = 0; k < kiosks; k++) {
				if(FrameIsOccupied(kd[k].events, (normalVid->Frame()+1))) {
					labels[countLabel] = 1;
				}
				else {
				labels[countLabel] = 0;
				}
				vcl_vector<float> temp = ExtractFeaturePerPixelCrowd(kd[k].ROI[0], kd[k].ROI[1], kd[k].ROI[2], kd[k].ROI[3], perPixelCrowd, hDiv, wDiv, im);
				if(featureSize == 0) {
					featureSize = (int)temp.size();
					for(int i = 0; i < kiosks*frames; i++) {
						features[i] = new float[featureSize];
					}
				}
 				featuresVector.push_back(temp);
				countLabel++;
			}
		}

		std::cout << "Converting feature vector to Array" << std::endl;
		//
		// convert FeatureVector to Array for integration
		//
		for(int i = 0; i < kiosks*frames; i++) {
			for(int k = 0; k < featureSize; k++) {
				features[i][k] = featuresVector[i][k];
				if (features[i][k] != 0) {
					std::cout << "Non zero feature\n" << std::endl;
				}
				else {
					std::cout << "Zero feature\n" << std::endl;
				}
			}
		}
		//
		// creat the matrices to train the SVM
		//
		Mat trainingDataMat((kiosks*frames), featureSize, CV_32FC1, features);
		Mat labelsMat((kiosks*frames),1, CV_32FC1, labels);
		//
		// Set up SVM's parameters
		//
		CvSVMParams params;
		params.svm_type    = CvSVM::C_SVC;
		params.kernel_type = CvSVM::RBF;
		params.term_crit   = cvTermCriteria(CV_TERMCRIT_ITER, 100, 1e-6);
		params.nu =1;
		std::cout << "Training SVM" << std::endl;
		//
		// Create and Train the SVM with parameters just created
		//
		CvSVM SVM;
		SVM.train(trainingDataMat, labelsMat, Mat(), Mat(), params);
		//
		// Save the SVM
		///home/webmaster/saivt-vxl/bin/qutapps
		SVM.save("/home/webmaster/saivt-vxl/qut/qutapps/qutSECAnalytics/models/QAL/SVM");
	} else {
		CvSVM SVM;
		SVM.load("/home/webmaster/saivt-vxl/qut/qutapps/qutSECAnalytics/models/QAL/SVM");
		//
		// load videos
		//
		qut_video_read<unsigned char>* normalVid =  new qut_video_read<unsigned char>(config, "/KioskData/qut-video-normal");
		qut_video_read<unsigned char>* motionVid =  new qut_video_read<unsigned char>(config, "/KioskData/qut-video-motion");
		int countLabel = 0;
		int countFrame = config->getInteger("/KioskData/TestFramesStart/@start");
		int endFrame = config->getInteger("/KioskData/TestFramesStart/@end");
		float frames = endFrame - countFrame;
		//
		// go to the designated frame
		//
		normalVid->GoTo(countFrame);
		motionVid->GoTo(countFrame);
		//
		// go through frames and test the svm
		//

		// pretty markup shit
		qut_video_write<unsigned char> outputvideo("detections.avi", normalVid->Current()->ni(), normalVid->Current()->nj(), true);
		// end pretty markup shit

		while((normalVid->Next() && (normalVid->Frame() <= endFrame)) && (motionVid->Next() && (motionVid->Frame() <= endFrame))) {
			//
			// lead features at each frame
			//
			vil_image_view<unsigned char> im = *normalVid->Current();
			vil_image_view<unsigned char> motion = *motionVid->Current();
			cc->Process(im, &motion);
			vil_image_view<unsigned int> blobID = cc->GetBlobIDMap();
			vil_image_view<float> densityMap = cc->GetDensityMap();
			vil_image_view<double> perPixelCrowd;

			// some pretty markup stuff
			vil_cv_image_view<unsigned char> markup;
			markup.deep_copy(im);
			// end pretty markup stuff

			PerPixelCrowdDensity(blobID, densityMap, cc->GetBlobEstimates()[0], perPixelCrowd);
			std::cout << "got estimate"<< std::endl;
			for(int k = 0; k < kiosks; k++) {
				std::cout << "extracting feature "<< std::endl;
				vcl_vector<float> temp = ExtractFeaturePerPixelCrowd(kd[k].ROI[0], kd[k].ROI[1], kd[k].ROI[2], kd[k].ROI[3], perPixelCrowd, hDiv, wDiv, im);
				float* testData;
				if(featureSize == 0) {
					featureSize = (int)temp.size();
					for(int i = 0; i < kiosks*frames; i++) {
						testData = new float[featureSize];
					}
				}

				for(int j = 0; j < featureSize; j++) {
					testData[j] = temp[j];
				}
				std::cout << "Creating sample vector "<< std::endl;
				Mat sample(1, featureSize, CV_32FC1, testData);
				std::cout << "Predicting "<< std::endl;
				float result = SVM.predict(sample);
				std::cout << "Kisok: " << k+1 << " Prediction: " << result << std::endl;
				//
				// Calculate Metrics
				//
				if(FrameIsOccupied(kd[k].testEvents, (normalVid->Frame()+1)) && result == 1) {
					truePositives++;
				}
				else if(FrameIsOccupied(kd[k].testEvents, (normalVid->Frame()+1)) && result == 0){
					falseNegatives++;
				}
				else if(!FrameIsOccupied(kd[k].testEvents, (normalVid->Frame()+1)) && result == 0) {
					trueNegatives++;
				}
				else if(!FrameIsOccupied(kd[k].testEvents, (normalVid->Frame()+1)) && result == 1) {
					falsePositives++;
				}
				// pretty markup shit
				if (result > 0)
				{
					for (int x = kd[k].ROI[0]; x < (kd[k].ROI[0] + kd[k].ROI[3]); x++)
					{
						for (int y = kd[k].ROI[1]; y < (kd[k].ROI[1] + kd[k].ROI[2]); y++)
						{
							markup(x, y, 1) = vcl_min(markup(x, y, 1) + 50, 255);
						}
					}
				}
				// end pretty markup shit
			}
			std::cout << "Frame: " << normalVid->Frame()+1 << std::endl;

			// pretty markup shit
			outputvideo.Write(markup);
			// end pretty markup shit
		}

		std::cout << std::fixed;
		std::cout << std::setprecision(2);
		std::cout << "truePositives: " << (truePositives/(frames*kiosks))*100 << "%" << std::endl;
		std::cout << "trueNegatives " << (trueNegatives/(frames*kiosks))*100 << "%" << std::endl;
		std::cout << "falsePositives: " << (falsePositives/(frames*kiosks))*100 << "%" << std::endl;
		std::cout << "falseNegatives: " << (falseNegatives/(frames*kiosks))*100 << "%" << std::endl;
		// pretty markup
		outputvideo.Close();
		// end pretty markup
	}
}

vcl_vector<float> ExtractFeaturePerPixelCrowd(int x, int y, int height, int width, vil_image_view<double> perPixelCrowd, int numHDivs, int numVDivs, vil_image_view<unsigned char> im) {
	int h = height/numHDivs;
	int w = width/numVDivs;
	vcl_vector<float> features;
	int count = 0;
	vil_image_view<double> image_ROI = vil_crop(perPixelCrowd, x, width, y, height);
	vil_image_view<unsigned char> original_ROI = vil_crop(im, x, width, y, height);
	vil_image_view<unsigned char> im_grey;
	vil_convert_planes_to_grey(original_ROI, im_grey);

	vil_image_view<unsigned char> orient_im;

	int cutoffs[4];
	cutoffs[0] = 45; cutoffs[1] = 90; cutoffs[2] = 135; cutoffs[3] = 180;

	vil_image_view<float> grad_mag;
	vil_orientations_from_sobel(im_grey, orient_im, grad_mag, 2 * HISTOGRAM_BINS);
	for(int i = 0; i < image_ROI.ni(); i +=w) {
		for(int j = 0; j < image_ROI.nj(); j += h) {
			float value = 0;
			for(int ii = i; ii < (i + w) && ii < image_ROI.ni(); ii++) {
				for(int jj = j; jj < (j + h) && jj < image_ROI.nj(); jj++) {
					value += (float)image_ROI(ii, jj);
				}
			}
			features.push_back(value);
			value = 0;
			double histvalues[4];
			histvalues[0] = 0; histvalues[1] = 0; histvalues[2] = 0; histvalues[3] = 0;
			count = 0;
			float gradMag = 0;;
			for(int ii = i; ii < (i + w) && ii < image_ROI.ni(); ii++) {
				for(int jj = j; jj < (j + h) && jj < image_ROI.nj(); jj++) {
					count++;

					histvalues[orient_im(ii, jj) % 4]++;
				}
			}
			features.push_back(histvalues[1]);
			features.push_back(histvalues[2]);
			features.push_back(histvalues[3]);
		}
	}

	return features;
}

bool FrameIsOccupied(vcl_vector<vcl_pair<int, int> > events, int frame) {
	for(int i = 0; i < events.size(); i++) {
		if(frame >= events[i].first && frame <= events[i].second)
			return true;
	}
	return false;
}
