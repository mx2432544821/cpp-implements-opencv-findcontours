#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

string file_name = "C:\\Pictures\\Saved Pictures\\test.jpg";


/**
 * 递归跟踪边缘
 * @param src 需要跟踪的图片，二值图
 * @param centrePoint 以这个点为中心开始寻找
 * @param startPoint 逆时针旋转的起点
 * @param contour 保存边缘的点
*/
void findNextPoint(Mat& src, Point centrePoint, Point startPoint, vector<Point>& contour, int& nbd);


/**
 * 寻找相同元素在顺时针或逆时针方向的索引值
 * @param point 旋转起点减去中心点
 * @param direction 旋转的方向，1代表顺时针，2代表逆时针
 *
 * @return 索引值
 */
int getTheSameElementIndex(Point point, int direction);


/**
 * 查找轮廓
 * @param src 经过二值化后的图像
 * @param contours 保存轮廓点
 */
void myFindContours(Mat& src, vector<vector<Point>>& contours);



/**
 * 为调用轮廓跟踪函数做一些准备工作，并调用轮廓跟踪函数
 * @param src 输入图像，是一个CV_32SC1类型的图像
 * @param contours 保存轮廓的向量
 * @param nbd 表示当前跟踪边界的序号
 * @param centrePoint 逐行扫描得到的第一个非零点
 * @param startPoint 根据非零点顺时针查找得到的地一个非零点
 */
void callFindNextPoint(Mat& src, vector<vector<Point>>& contours, int& nbd, Point centrePoint, Point startPoint);



/**
 * 顺时针旋转，以中心像素的左边一个像素开始
*/

static int clockWiseArray[8][2] = { {0, -1},
    {-1, -1},
    {-1, 0},
    {-1, 1},
    {0, 1},
    {1, 1},
    {1, 0},
    {1, -1},
};

/**
 * 逆时针旋转数组，以中心点的左边像素为起始点
*/

static int antiClockWiseArray[8][2] = { {0, -1},
    {1, -1},
    {1, 0},
    {1, 1},
    {0, 1},
    {-1, 1},
    {-1, 0},
    {-1, -1},
};



//从这里开始
int main(int argc, char** argv)
{
    Mat img = imread(file_name, 0);
    vector<vector<Point>> contours;
    Mat thresh;
    Mat test = Mat::zeros(img.size(), img.type());
    threshold(img, thresh, 0, 255, THRESH_OTSU | THRESH_BINARY_INV);
    myFindContours(thresh, contours);
    cout << contours.size() << endl;
    for (vector<Point> contour : contours)
        for (Point point : contour)
        {
            test.at<uchar>(point) = 255;
            //cout << contour.size() << endl;
        }
                
    Mat result = Mat::zeros(img.size(), img.type());
    vector<vector<Point>> con;
    findContours(thresh, con, RETR_LIST, CHAIN_APPROX_NONE);
    cout << con.size() << endl;
    for (vector<Point> vec : con)
        for (Point point : vec)
        {
            result.at<uchar>(point) = 255;
            //cout << vec.size() << endl;
        }
    //for (int ix = 0; ix < contours.size(); ix++)
        //cout << "std contour.size() = " << con[ix].size() << "\tmy contour.size() = " << contours[ix].size() << endl;
            
    cout << "img.size() = " << img.size() << endl;
    Mat distance;
    subtract(result, test, distance);
    imshow("origin", img);
    imshow("thresh", thresh);
    imshow("contours", test);
    imshow("result", result);
    imshow("distance", distance);
    waitKey();
    return 0;
}



void myFindContours(Mat& src, vector<vector<Point>>& contours)
{
    Mat input = Mat::zeros(src.rows + 2, src.cols + 2, CV_32SC1);
    for (int ix = 1; ix < input.rows - 1; ix++)
        for (int jx = 1; jx < input.cols - 1; jx++)
            if (src.at<uchar>(ix - 1, jx - 1) == 255)
                input.at<int>(ix, jx) = 1;
    int nbd = 1;
    int startIndex = 0;
    bool checked = false;
    for (int ix = 1; ix < input.rows - 1; ix++)
    {
        for (int jx = 1; jx < input.cols - 1; jx++)
        {
            checked = false;
            if (input.at<int>(ix, jx) == 1 && input.at<int>(ix, jx - 1) == 0)
                startIndex = 0;
            else if (input.at<int>(ix, jx) >= 1 && input.at<int>(ix, jx + 1) == 0)
                startIndex = 4;
            else
                continue;
            //获取当前边界的父边界
            for (int i = startIndex; i < startIndex + 8; i++)
                if (input.at<int>(ix + clockWiseArray[i][0], jx + clockWiseArray[i][1]) != 0)
                {
                    Point centrePoint(jx, ix);
                    Point startPoint(centrePoint.x + clockWiseArray[i][1], centrePoint.y + clockWiseArray[i][0]);
                    callFindNextPoint(input, contours, nbd, centrePoint, startPoint);
                    checked = true;
                    break;
                }
            //处理单个像素点
            if (!checked)
            {
                input.at<int>(ix, jx) = nbd++;
                vector<Point> contour;
                Point found(jx - 1, ix - 1);
                contour.push_back(found);
                contours.push_back(contour);
            }
        }
    }
}

void findNextPoint(Mat& src, Point centrePoint, Point startPoint, vector<Point>& contour, int& nbd)
{
    if (contour.size() > 2)
    {
        if (contour[1].x + 1 == centrePoint.x && contour[1].y + 1 == centrePoint.y && contour[0].x + 1 == startPoint.x && contour[0].y + 1 == startPoint.y)
            return;
    }
    Point point(startPoint.x - centrePoint.x, startPoint.y - centrePoint.y);
    int sameIndex = getTheSameElementIndex(point, 2);

    //逆时针查找非零像素
    int index = sameIndex + 1;
    int pixel = 0;//当前像素的值
    int row = centrePoint.y, col = centrePoint.x;

    for (int ix = 0; ix < 8; ix++, index++)
    {
        pixel = src.at<int>(row + antiClockWiseArray[index % 8][0], col + antiClockWiseArray[index % 8][1]);
        bool checked = false;


        if (index % 8 == 4 && src.at<int>(row, col + 1) == 0)
        {
            src.at<int>(row, col) = -nbd;
            checked = true;
        }
        if (pixel)
        {
            Point foundPoint(col + antiClockWiseArray[index % 8][1], row + antiClockWiseArray[index % 8][0]);
            if (!checked && src.at<int>(row, col) == 1)
                src.at<int>(row, col) = nbd;
            Point point(foundPoint.x - 1, foundPoint.y - 1);
            contour.push_back(point);
            findNextPoint(src, foundPoint, centrePoint, contour, nbd);
            break;
        }
    }
}

int getTheSameElementIndex(Point point, int direction) {
    int* p_array = nullptr;
    switch (direction) {
    case 1:
        p_array = &clockWiseArray[0][0];
        break;
    case 2:
        p_array = &antiClockWiseArray[0][0];
        break;
    default:
        cout << "Direction out of bound!" << endl;
        p_array = &clockWiseArray[0][0];
        break;
    }

    for (int ix = 0; ix < 8; ix++) {
        if (point.y == *p_array && point.x == *(p_array + 1))
            return ix;
        p_array += 2;
    }
    return -111;
}


void callFindNextPoint(Mat& src, vector<vector<Point>>& contours, int& nbd, Point centrePoint, Point startPoint)
{
    nbd++;
    vector<Point> contour;
    Point point(centrePoint.x - 1, centrePoint.y - 1);
    contour.push_back(point);
    findNextPoint(src, centrePoint, startPoint, contour, nbd);
    //由于采用尾递归，在判断达到轮廓跟踪起点时会在尾部多插入两个重复的点，利用pop_back()将这两个点去掉
    contour.pop_back();
    contour.pop_back();
    contours.push_back(contour);
}
