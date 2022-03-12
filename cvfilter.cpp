#include <QDebug>
#include <fstream>
#include <QFile>
#include <QTemporaryFile>
#include <QAndroidJniEnvironment>
#include <QAndroidJniObject>
#include "cvfilter.h"

CVFilter::CVFilter(QObject *parent) : QAbstractVideoFilter(parent)
{
    qDebug()<<"hey terah";
   // connect(&tUpdate, &QTimer::timeout, this, &VideoStreamer::streamVideo);
    QAndroidJniObject mediaDir = QAndroidJniObject::callStaticObjectMethod("android/os/Environment", "getExternalStorageDirectory", "()Ljava/io/File;");
    QAndroidJniObject mediaPath = mediaDir.callObjectMethod("getAbsolutePath", "()Ljava/lang/String;");
    QAndroidJniObject activity = QAndroidJniObject::callStaticObjectMethod("org/qtproject/qt5/android/QtNative", "activity", "()Landroid/app/Activity;");
    QString ModelPath = mediaPath.toString()+"/models/frozen_inference_graph_V2.pb";
    std::string MP = ModelPath.toStdString();
    QString ConfigPath = mediaPath.toString()+"/models/ssd_mobilenet_v2_coco_2018_03_29.pbtxt";
    std::string CP = ConfigPath.toStdString();
    QString LabelPath = mediaPath.toString()+"/models/COCO_labels.txt";
    std::string LP = LabelPath.toStdString();

    net = dnn::readNetFromTensorflow(MP, CP);
    qDebug()<<"importation working";

    if(net.empty())
    {
        qDebug()<<"init the model net error";
        exit(-1);
    }
    bool result = getFileContent(LP);
    if(!result)
    {
        qDebug()<<"loading labels failed";
        exit(-1);
    }
//    QFile xml(":/assets/classifiers/lbpcascade_frontalface.xml");

 //   if(xml.open(QFile::ReadOnly | QFile::Text))
//    {
   //     QTemporaryFile temp;
     //   if(temp.open())
       // {
//            temp.write(xml.readAll());
  //          temp.close();
    //        if(classifier.load(temp.fileName().toStdString()))
      //      {
        //        qDebug() << "Successfully loaded classifier!";
          //  }
//            else
  //          {
    //            qDebug() << "Could not load classifier.";
      //      }
        //}
        //else
        //{
         //   qDebug() << "Can't open temp file.";
        //}
    //}
    //e/lse
    //{
      //  qDebug() << "Can't open XML.";
    //}

}

CVFilter::~CVFilter()
{
    if(!processThread.isFinished()) {
        processThread.cancel();
        processThread.waitForFinished();
    }
}

QVideoFilterRunnable *CVFilter::createFilterRunnable()
{
    return new CVFilterRunnable(this);
}

void CVFilter::registerQMLType()
{
    qmlRegisterType<CVFilter>("CVFilter", 1, 0, "CVFilter");
}

bool CVFilter::getFileContent(std::string fileName)
{
    std::ifstream in(fileName.c_str());
    if(!in.is_open()) return false;
    std::string str;
    while(std::getline(in, str))
    {
        if(str.size()>0) Names.push_back(str);
    }

    in.close();
    return true;
}



QImage CVFilter::videoFrameToImage(QVideoFrame *frame)
{
    if(frame->handleType() == QAbstractVideoBuffer::NoHandle){

        QImage image = qt_imageFromVideoFrame(*frame);

        if(image.isNull()){
            qDebug() << "-- null image from qt_imageFromVideoFrame";
            return QImage();
        }

        if(image.format() != QImage::Format_RGB32){
            image = image.convertToFormat(QImage::Format_RGB32);
        }

        return image;
    }

    if(frame->handleType() == QAbstractVideoBuffer::GLTextureHandle){
        QImage image(frame->width(), frame->height(), QImage::Format_RGB32);
        GLuint textureId = frame->handle().toUInt();//static_cast<GLuint>(frame.handle().toInt());
        QOpenGLContext *ctx = QOpenGLContext::currentContext();
        QOpenGLFunctions *f = ctx->functions();
        GLuint fbo;
        f->glGenFramebuffers(1,&fbo);
        GLint prevFbo;
        f->glGetIntegerv(GL_FRAMEBUFFER_BINDING,&prevFbo);
        f->glBindFramebuffer(GL_FRAMEBUFFER,fbo);
        f->glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, textureId, 0);
        f->glReadPixels(0, 0, frame->width(), frame->height(), GL_RGBA, GL_UNSIGNED_BYTE, image.bits());
        f->glBindFramebuffer(GL_FRAMEBUFFER, static_cast<GLuint>(prevFbo));
        return image.rgbSwapped();
    }

    qDebug() << "-- Invalid image format...";
    return QImage();
}

CVFilterRunnable::CVFilterRunnable(CVFilter *filter) : QObject(nullptr), filter(filter)
{

}

CVFilterRunnable::~CVFilterRunnable()
{
    filter = nullptr;
}

QVideoFrame CVFilterRunnable::run(QVideoFrame *input, const QVideoSurfaceFormat &surfaceFormat, QVideoFilterRunnable::RunFlags flags)
{
    Q_UNUSED(surfaceFormat);
    Q_UNUSED(flags);

    if(!input || !input->isValid()){
        return QVideoFrame();
    }

    if(filter->isProcessing){
        return * input;
    }

    if(!filter->processThread.isFinished()){
        return * input;
    }

    filter->isProcessing = true;

    QImage image = filter->videoFrameToImage(input);

    // All processing has to happen in another thread, as we are now in the UI thread.
    filter->processThread = QtConcurrent::run(this, &CVFilterRunnable::processImage, image);

    return * input;
}

void CVFilterRunnable::processImage(QImage &image)
{

    //if android, make image upright
#ifdef Q_OS_ANDROID
    QPoint center = image.rect().center();
    QMatrix matrix;
    matrix.translate(center.x(), center.y());
    matrix.rotate(90);
    image = image.transformed(matrix);
#endif

    if(!image.isNull()){
        detect(image);
    }

//    QString filename = QStandardPaths::writableLocation(QStandardPaths::PicturesLocation) + "/" + "my_image.png";

//    if(!QFile::exists(filename)){
//        image.save(filename);
//    }

}




void CVFilterRunnable::detect(QImage image)
{


    image = image.convertToFormat(QImage::Format_RGB888);
    cv::Mat src(image.height(),
                image.width(),
                CV_8UC3,
                image.bits(),
                image.bytesPerLine());


    Mat blobimg = dnn::blobFromImage(src, 1.0, Size(300,300), 0.0, true);

    filter->net.setInput(blobimg);

    Mat detection = filter->net.forward("detection_out");

    Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());

    QJsonArray rects;
    QJsonObject rect;

    double rX, rY, rWidth, rHeight;

    Size frameSize = src.size();
    std::vector<Rect> detected;

    const float confidence_threshold = 0.25;
    for(int i=0; i < detectionMat.rows; i++)
    {
        float detect_confidence = detectionMat.at<float>(i, 2);

        if(detect_confidence > confidence_threshold)
        {
            size_t det_index = (size_t)detectionMat.at<float>(i, 1);
            float x1 = detectionMat.at<float>(i, 3)*src.cols;
            float y1 = detectionMat.at<float>(i, 4)*src.rows;
            float x2 = detectionMat.at<float>(i, 5)*src.cols;
            float y2 = detectionMat.at<float>(i, 6)*src.cols;
            Rect rec((int)x1, (int)y1, (int)(x2 - x1), (int)(y2 - y1));
            qDebug()<<filter->Names[det_index].c_str();
            detected.push_back(rec);
        }
    }

    //cv::flip(frame, frame, 0);

//    Mat frameGray;
//
//    if(frame.channels() == 3){
//        cvtColor( frame, frameGray, COLOR_BGR2GRAY );
//    }else if(frame.channels() == 4) {
 //       cvtColor( frame, frameGray, COLOR_BGRA2GRAY );
 //   }

//    equalizeHist( frameGray, frameGray );

 //   std::vector<cv::Rect> detected;

    //resize the frame
//    double imageWidth = image.size().width();
//    double imageHeight = image.size().height();

//    double resizedWidth = 320;
//    double resizedHeight = (imageHeight/imageWidth) * resizedWidth;

//    cv::resize(frameGray, frameGray, cv::Size((int)resizedWidth, (int)resizedHeight));

//    filter->classifier.detectMultiScale(frameGray, detected, 1.1, 10);






    for(size_t i = 0; i < detected.size(); i++){

        rX = double(detected[i].x) / double(frameSize.width);
        rY = double(detected[i].y) / double(frameSize.height);
        rWidth = double(detected[i].width) / double(frameSize.width);
        rHeight = double(detected[i].height) / double(frameSize.height);

        Point center( detected[i].x + detected[i].width/2, detected[i].y + detected[i].height/2 );
        ellipse( src, center, Size( detected[i].width/2, detected[i].height/2 ), 0, 0, 360, Scalar( 255, 0, 255 ), 4 );

        qDebug() << rX << rY << rWidth << rHeight;

        rect.insert("rX",rX);
        rect.insert("rY",rY);
        rect.insert("rWidth",rWidth);
        rect.insert("rHeight",rHeight);

        rects.append(rect);
    }

    //qDebug() << "Count: " << detected.size();

    if(rects.count() > 0){
        emit filter->objectDetected(QString::fromStdString(QJsonDocument(rects).toJson().toStdString()));
    }

    //saving processed image to disk after every 5 sec
//    qint64 now = QDateTime::currentDateTime().toSecsSinceEpoch();
//    qint64 diff = now - filter->lastProcessedImageAt;

//    if(diff >= 5){

//        cvtColor(frameGray, frameGray, COLOR_BGR2RGB);
//        QImage processedImage((uchar*)frameGray.data,frameGray.cols,frameGray.rows,QImage::Format_RGB888);

//        QString filename = QStandardPaths::writableLocation(QStandardPaths::PicturesLocation) + "/" + "processed_image_" + QString::number(now) + ".png";

//        processedImage.save(filename);

//        filter->lastProcessedImageAt = now;
//    }

    filter->isProcessing = false;

}

