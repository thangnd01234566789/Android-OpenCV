package com.example.facedetected;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Context;
import android.os.Bundle;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.JavaCameraView;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;

public class MainActivity extends AppCompatActivity implements CameraBridgeViewBase.CvCameraViewListener2 {

    JavaCameraView javaCameraView;
//    BaseLoaderCallback baseCallBack;
    File casFile;
    CascadeClassifier faceDetector;
    private Mat mRgb, mGray;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
//        OpenCVLoader.initDebug();

        javaCameraView = (JavaCameraView) findViewById(R.id.javaCameraView);

        if (!OpenCVLoader.initDebug()){
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_4_0, this, baseCallBack);
        }
        else {
            try {
                baseCallBack.onManagerConnected(LoaderCallbackInterface.SUCCESS);
            } catch (IOException e) {
                e.printStackTrace();
            }
        }

        javaCameraView.setCvCameraViewListener(this);
    }

    @Override
    public void onCameraViewStarted(int width, int height) {
        mRgb = new Mat();
        mGray = new Mat();

    }

    @Override
    public void onCameraViewStopped() {
        mRgb.release();
        mGray.release();
    }

    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        mRgb = inputFrame.rgba();
        mGray = inputFrame.gray();

        MatOfRect faceDetections = new MatOfRect();
        faceDetector.detectMultiScale(mRgb, faceDetections);

        for(Rect rect: faceDetections.toArray()){
            Imgproc.rectangle(mRgb, new Point(rect.x, rect.y), new Point(rect.x + rect.width, rect.y + rect.height), new Scalar(255, 0 , 255));
        }

        return mRgb;
    }

    private BaseLoaderCallback baseCallBack = new BaseLoaderCallback() {
        @Override
        public void onManagerConnected(int status) throws IOException {
            super.onManagerConnected(status);
            switch (status){
                case BaseLoaderCallback.SUCCESS:
                {
                    InputStream is = getResources().openRawResource(R.raw.haarcascade_frontalface_alt2);
                    File cascadeDir = getDir("cascade", Context.MODE_PRIVATE);
                    casFile = new File(cascadeDir, "haarcascade_frontalface_alt2.xml");

                    FileOutputStream fos = new FileOutputStream(casFile);

                    byte[] buffer = new byte[4096];
                    int bytesRead;

                    while ((bytesRead = is.read(buffer)) != 0){
                        fos.write(buffer, 0 ,bytesRead);
                    }
                    is.close();
                    fos.close();

                    faceDetector = new CascadeClassifier(casFile.getAbsolutePath());

                    if (faceDetector.empty()){
                        faceDetector = null;
                    }
                    else {
                        cascadeDir.delete();
                    }
                }
                javaCameraView.enableView();
                break;
                default:
                {
                    super.onManagerConnected(status);
                }
                break;
            }
        }
    };
}