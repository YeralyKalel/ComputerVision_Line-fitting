#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"

#include <time.h>
#include <stdlib.h>

#include <iostream>
#include <math.h>

size_t GetIterationNumber(
    const double& inlierRatio_,
    const double& confidence_,
    const size_t& sampleSize_)
{
    std::cout << "Inlier ratio is " << inlierRatio_ << std::endl;
    double a =
        log(1.0 - confidence_);
    double b =
        log(1.0 - std::pow(inlierRatio_, sampleSize_));

    if (abs(b) < std::numeric_limits<double>::epsilon())
        return std::numeric_limits<size_t>::max();

    return a / b;
}

void GetEdgePoints(cv::Mat &image, cv::Mat &edges, std::vector<cv::Point2d> &points) {
    //cv::Mat edges;
    cv::Canny(image, edges, 80, 160);
    //cv::imshow("Original", image);
    //cv::imshow("Edges", edges);

    cv::Mat binaryMat(edges.size(), edges.type());
    cv::threshold(edges, binaryMat, 100, 255, cv::THRESH_BINARY);

    uint8_t* binaryMatData = binaryMat.data;
    int width = binaryMat.cols;
    int height = binaryMat.rows;
    int _stride = binaryMat.step;
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            if (binaryMatData[i * _stride + j]) {
                points.push_back(cv::Point2d(j, i));
            }
        }
    }
}

std::vector<int> RandomPerm(int sampleSize, int dataSize) {
    std::vector<int> result;
    for (size_t sampleIdx = 0; sampleIdx < sampleSize && sampleIdx < dataSize; sampleIdx++) {
        int val;
        bool isFound;
        do {
            val = rand() % (dataSize - 1);
            isFound = false;
            for (size_t i = 0; i < result.size(); i++) {
                if (val == result[i]) {
                    isFound = true;
                    break;
                }
            }
        } while (isFound);
        result.push_back(val);
    }
    return result;
}

struct Line {
public:
    double a = 0, b = 0, c = 0;
};

void DrawPoints(const std::vector<cv::Point2d> &points, cv::Mat image) {
    for (int i = 0; i < points.size(); i++) {
        circle(image, points[i], 0, cv::Scalar(255, 255, 255), -1);
    }
}

double DistanceBetweenTwoPoints(cv::Point2d p1, cv::Point2d p2) {
    cv::Point2d p =  p1 - p2;
    return pow(p.x, 2) + pow(p.y, 2);
}

void RANSAC(const std::vector<cv::Point2d> &dataset, std::vector<int> &bestInliersIdx, Line &bestModel, const int kSampleSize, double threshold, double confidence, int maxIter, cv::Mat& image, bool isDrawing = false) {
    
    if (dataset.size() < kSampleSize) {
        return;
    }
    
    bestInliersIdx.clear();

    int iter = 0;
    int maxIter_ = maxIter;
    std::vector<int> inliersIdx;

    std::vector<int> sampleIdx(kSampleSize);
    std::vector<cv::Point2d> sample(kSampleSize);
    cv::Mat lineModel;

    cv::Mat imageLine = image.clone();
    cv::Mat imagePoint;
    while (iter++ < maxIter_) {
        //Select random points
        sample.clear();
        sampleIdx = RandomPerm(kSampleSize, dataset.size());
        for (size_t i = 0; i < kSampleSize; i++) {
            sample.push_back(dataset[sampleIdx[i]]);
        }

        //Create model:
        cv::Point2d p1 = sample[0];
        cv::Point2d p2 = sample[1];
        cv::Point2d v = p2 - p1;
        v = v / cv::norm(v);
        //Rotate v by 90 degrees to get n
        cv::Point2d n;
        n.x = -v.y;
        n.y = v.x;

        double a = n.x;
        double b = n.y;
        double c = -(a * p1.x + b * p1.y);

        if (isDrawing) {
            imagePoint = imageLine.clone();
            circle(imagePoint, p1, 3, cv::Scalar(0, 255, 0), -1);
            circle(imagePoint, p2, 3, cv::Scalar(0, 255, 0), -1);
            cv::imshow("RANSAC", imagePoint);
            cv::waitKey(1);
        }

        //Fit model:
        cv::Point2d point;
        double distance;
        inliersIdx.clear();
        for (size_t dataIdx = 0; dataIdx < dataset.size(); dataIdx++) {
            point = dataset[dataIdx];
            distance = abs(a * point.x + b * point.y + c);
            if (distance < threshold) {
                inliersIdx.push_back(dataIdx);
            }
        }

        if (inliersIdx.size() > bestInliersIdx.size() && inliersIdx.size() > 200) {
            bestModel.a = a;
            bestModel.b = b;
            bestModel.c = c;
            bestInliersIdx = inliersIdx;

            // Update the maximum iteration number
            maxIter_ = GetIterationNumber(
                static_cast<double>(bestInliersIdx.size()) / static_cast<double>(dataset.size()),
                confidence,
                kSampleSize);
            
            printf("Inlier number = %d\tMax iterations = %d\n", (int)bestInliersIdx.size(), maxIter_);

            if (isDrawing) {
                imageLine = image.clone();
                for (int i = 0; i < bestInliersIdx.size(); i++) {
                    circle(imageLine, dataset[bestInliersIdx[i]], 0, cv::Scalar(255, 0, 0), -1);
                }
                circle(imageLine, p1, 3, cv::Scalar(0, 255, 0), -1);
                circle(imageLine, p2, 3, cv::Scalar(0, 255, 0), -1);
                cv::imshow("RANSAC", imageLine);
                cv::waitKey(1);
            }
        }
        std::cout << (double)iter / maxIter_ * 100 << "%\r";
        std::cout.flush();
    }

    //Draw best line:
    cv::line(image,
				cv::Point2d(0, -bestModel.c / bestModel.b),
				cv::Point2d(image.cols, (-bestModel.a * image.cols - bestModel.c) / bestModel.b),
				cv::Scalar(0, 0, 255),
				2);
    cv::imshow("RANSAC", image);
    cv::waitKey(1);

}

int main(int argc, char** argv)
{
    //0095
    //0097
    //left
    //right
    srand(time(NULL));
    std::string image_path = cv::samples::findFile("Images/0095.jpg"); 
    cv::Mat img = imread(image_path, cv::IMREAD_COLOR);
    cv::Mat emptyImg(img.size(), img.type(), cv::Scalar(0,0,0));

    std::vector<cv::Point2d> points;
    cv::Mat edges_image;
    GetEdgePoints(img, edges_image, points);
    Line line;
    std::cout << points.size() << std::endl;
    DrawPoints(points, emptyImg);
    cv::imshow("RANSAC", emptyImg);
    //cv::imshow("RANSAC", img);

    std::vector<int> inliersIdx;
    for (int i = 0; i < 50; i++) {
        //algorithm is not using confidence, but is here (its implementation is commented inside the function)
        RANSAC(points, inliersIdx, line, 2, 2.0, 0.99, 100, emptyImg, false);
        std::cout << "At step " << i + 1 << " Line was found with size " << inliersIdx.size() << std::endl;
    }
    std::cout << "Finished" << std::endl;
    int k = cv::waitKey(0);

    return 0;
}
