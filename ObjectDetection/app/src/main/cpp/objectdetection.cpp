#include <jni.h>
#include <string>
#include <opencv2/imgproc/types_c.h>
#include "include/opencv2/wechat_qrcode.hpp"
#include "bitmap_utils.h"
#include "android/log.h"
#include <jni.h>


using namespace cv;
using namespace std;

#define LOG_TAG "System.out"
#define  LOGI(...)  __android_log_print(ANDROID_LOG_INFO,LOG_TAG,__VA_ARGS__)
#define  LOGD(...)  __android_log_print(ANDROID_LOG_DEBUG,LOG_TAG,__VA_ARGS__)
#define  LOGE(...)  __android_log_print(ANDROID_LOG_ERROR,LOG_TAG,__VA_ARGS__)

Ptr<wechat_qrcode::WeChatQRCode> detector;

Mat balance(Mat img)
{
    vector<Mat> bgrMat;
    split(img, bgrMat);

    vector<float> weighted;
    for(int i=0; i<bgrMat.size(); i++)
        weighted.push_back(mean(bgrMat[i])[0]);
    float k = (weighted[0] + weighted[1] + weighted[2])/3;
    for(int i=0; i<weighted.size();i++)
        weighted[i] = k / weighted[i];

//    vector<Mat> mergeMat;
    for(int i=0; i<weighted.size();i++)
    {
        addWeighted(bgrMat[i], weighted[i], bgrMat[i], 0, 0, bgrMat[i]);
    }
    Mat dstMat;
    merge(bgrMat, dstMat);
    return dstMat;
}
int color_class(Mat src, int x, int y, int w, int h)
{
    int redsum=0, yellowsum=0, blacksum=0;
    int r, g, b;
    int color_index=-1;
    for(int i=0; i<h; i++)
    {
        for(int j=0; j<w; j++)
        {
            r = src.at<Vec4b>(y+i, x+j)[0];
            g = src.at<Vec4b>(y+i, x+j)[1];
            b = src.at<Vec4b>(y+i, x+j)[2];
            if (r>1.5*g && r > 1.5*b && abs(g-b) < 30)
                redsum+=1;
            else if(abs(r-g)<30 && r > 2*b && g > 2*b)
                yellowsum+=1;
            else if(abs(r-b) < 20 && abs(r-g) < 20 && r < 70)
                blacksum+=1;
        }
    }
    std::cout << "r,y,b" << redsum << ", " << yellowsum << ", " << blacksum << std::endl;
    if (redsum > yellowsum && redsum > blacksum)
        color_index = 0;
    else if(yellowsum > redsum && yellowsum > blacksum)
        color_index = 1;
    else
        color_index = 2;
    return color_index;
}

extern "C"
JNIEXPORT jstring JNICALL
Java_org_pytorch_demo_objectdetection_MainActivity_opBitmap(JNIEnv *env, jobject thiz, jobject bitmap, jobject argb8888) {
    // TODO: implement opBitmap()
    Mat srcMat;// rgba
    vector<Mat> vPoints;
    vector<int> color;
    try{
        bitmap2Mat(env, bitmap, &srcMat, true);
    }
    catch (const std::exception& e)
    {
        std::cout << e.what() << std::endl;
        return env->NewStringUTF("format fail!");
    }

    if(srcMat.empty()){
        std::cout << "img empty: " << srcMat.empty() << std::endl;
        return env->NewStringUTF("empty Image");
    }
    vector<string> ret = detector->detectAndDecode(srcMat, vPoints);
    LOGI("decode over:");
    stringstream ss;
    ss << ret.size() << " ";
    for(int i=0; i<ret.size(); i++)
    {
        int x, y, xm, ym, w, h;
        x = min(min((int)vPoints[i].at<float>(0, 0), (int)vPoints[i].at<int>(1, 0)), (int)vPoints[i].at<int>(2, 0));
        y = min(min((int)vPoints[i].at<float>(0, 1), (int)vPoints[i].at<float>(1, 1)), (int)vPoints[i].at<float>(2, 1));

        xm = max(max((int)vPoints[i].at<float>(0, 0), (int)vPoints[i].at<float>(1, 0)), (int)vPoints[i].at<float>(2, 0));
        ym = max(max((int)vPoints[i].at<float>(0, 1), (int)vPoints[i].at<float>(1, 1)), (int)vPoints[i].at<float>(2, 1));

        h = ym - y;
        w = xm - x;

        int v = color_class(srcMat, x, y, w, h);
        color.push_back(v);
        ss << v << " " << ret[i] << " ";
    }
    string all_info = ss.str();
    if(ret.empty())
        return env->NewStringUTF("none");
    else
        return env->NewStringUTF(all_info.c_str());//使用dstMat创建一个Bitmap对象
}
extern "C"
JNIEXPORT jstring JNICALL
Java_org_pytorch_demo_objectdetection_MainActivity_initQr(JNIEnv *env, jobject thiz) {
    // TODO: implement initQr()
    try {
        detector = makePtr<wechat_qrcode::WeChatQRCode>(
                "/sdcard/qr_module/detect_prototxt",
                "/sdcard/qr_module/detect_caffemodel",
                "/sdcard/qr_module/sr_prototxt",
                "/sdcard/qr_module/sr_caffemodel");
    }
    catch (const std::exception& e)
    {
        std::cout << e.what() << std::endl;
        return env->NewStringUTF("fail load");
    }
    return env->NewStringUTF("sucess load");
}
extern "C"
JNIEXPORT jint JNICALL
Java_org_pytorch_demo_objectdetection_MainActivity_colorBitmap(JNIEnv *env, jobject thiz,
                                                               jobject bitmap,jint left, jint top, jint width, jint height, jobject argb8888) {
    // TODO: implement colorBitmap()
//    String color[] = {"red", "yellow", "green", "qing", "blue", "pin", "red", "black", "white"};
    int color_lower[9][3] = {{0, 35, 46},
                             {15, 35, 46},
                             {48, 35, 46},
                             {78, 35, 46},
                             {89, 35, 46},
                             {125, 35, 46},
                             {156, 35, 46},
                             {0, 0, 0},
                             {0, 0, 221}};
    int color_upper[9][3] = {{15, 255, 255},
                             {47, 255, 255},
                             {77, 255, 255},
                             {88, 255, 255},
                             {124, 255, 255},
                             {155, 255, 255},
                             {180, 255, 255},
                             {180, 255, 35},
                             {180, 30, 255}};

    Mat srcMat;// rgba
    Mat dstMat;// BGR

    bitmap2Mat(env, bitmap, &srcMat, true);
    cvtColor(srcMat, dstMat, CV_RGBA2BGR);

    Mat balanceMat = balance(dstMat);
    Mat blurMat;
    GaussianBlur(balanceMat, blurMat, Size(5,5), 0);

    Mat subMat = blurMat(Range(top, top+height), Range(left, left+width));
    Mat hsvMat;
    cvtColor(subMat, hsvMat, CV_BGR2HSV);

    vector<Mat> v_hsvMat;
    vector<int> max_hsv;
    split(hsvMat, v_hsvMat);

    int channels[] = {0};
    int histSize[] = {256};
    float Ranges[] = {0, 256};
    const float* hsvRanges[] = {Ranges};
    MatND dstHist;
    for (int i=0; i<3; i++){
        calcHist(&v_hsvMat[i], 1, channels, Mat(), dstHist, 1, histSize, hsvRanges, true, false);
        int maxIndex = 0;
        minMaxIdx(dstHist, 0, 0, 0, &maxIndex);
        max_hsv.push_back(maxIndex);
    }

    int color_index=-1;
    for(int i=0; i<9; i++){
        if(max_hsv[0] <= color_upper[i][0] && max_hsv[0] >= color_lower[i][0]
        && max_hsv[1] <= color_upper[i][1] && max_hsv[1] >= color_lower[i][1]
        && max_hsv[2] <= color_upper[i][2] && max_hsv[2] >= color_lower[i][1]) {
            color_index = i;
            break;
        }
    }
    return color_index;
}

extern "C"
JNIEXPORT jobjectArray JNICALL
Java_org_pytorch_demo_objectdetection_MainActivity_opBitmap2(JNIEnv *env, jobject thiz,
                                                             jobject bitmap, jobject argb8888) {
    // TODO: implement opBitmap2()
    Mat srcMat;// rgba
    vector<Mat> vPoints;
    vector<int> color;
    try{
        bitmap2Mat(env, bitmap, &srcMat, true);
    }
    catch (const std::exception& e)
    {
        std::cout << e.what() << std::endl;
//        return env->NewStringUTF("format fail!");
    }

    if(srcMat.empty()){
        std::cout << "img empty: " << srcMat.empty() << std::endl;
//        return env->NewStringUTF("empty Image");
    }
    vector<string> ret = detector->detectAndDecode(srcMat, vPoints);
    vector<string> color_cls;
    string c0 = "0";
    string c1 = "1";
    string c2 = "2";
    color_cls.push_back(c0);
    color_cls.push_back(c1);
    color_cls.push_back(c2);

    LOGI("decode over:");
    stringstream ss;
    ss << ret.size() << " ";
    jobjectArray ret_array = (jobjectArray)env->NewObjectArray(5,env->FindClass("java/lang/String"),env->NewStringUTF(""));
    for(int i=0; i<ret.size(); i++)
    {
        int x, y, xm, ym, w, h;
        x = min(min((int)vPoints[i].at<float>(0, 0), (int)vPoints[i].at<int>(1, 0)), (int)vPoints[i].at<int>(2, 0));
        y = min(min((int)vPoints[i].at<float>(0, 1), (int)vPoints[i].at<float>(1, 1)), (int)vPoints[i].at<float>(2, 1));

        xm = max(max((int)vPoints[i].at<float>(0, 0), (int)vPoints[i].at<float>(1, 0)), (int)vPoints[i].at<float>(2, 0));
        ym = max(max((int)vPoints[i].at<float>(0, 1), (int)vPoints[i].at<float>(1, 1)), (int)vPoints[i].at<float>(2, 1));

        h = ym - y;
        w = xm - x;

        int v = color_class(srcMat, x, y, w, h);
        color.push_back(v);
        ret[i] = color_cls[v] + " " + ret[i];
        env->SetObjectArrayElement(ret_array, i, env->NewStringUTF(ret[i].c_str()));
    }
    return ret_array;
}