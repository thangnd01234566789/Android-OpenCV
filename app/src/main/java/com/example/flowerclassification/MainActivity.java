package com.example.flowerclassification;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import android.annotation.SuppressLint;
import android.app.Activity;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.Camera;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Log;
import android.view.Surface;
import android.view.SurfaceView;
import android.view.View;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import org.checkerframework.common.reflection.qual.NewInstance;
import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraActivity;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.JavaCameraView;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;

import java.io.IOException;
import java.util.Collections;
import java.util.List;

public class MainActivity extends CameraActivity {
    private static final int CAMERA_REQUEST = 100;
    private static Recognition recognition;
    ImageView imageView;
    TextView predictionResults;
    Bitmap imgBitmap;
    Uri imgUri;
    private Mat mRgb;
    private CameraBridgeViewBase mOpenCVCameraView;
    private static String LOGTAG = "OpenCV_Log";
    private int index = 0;
    private  static int camera_index;

    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status){
                case LoaderCallbackInterface.SUCCESS:{
                    Log.v(LOGTAG, " OpenCV_Loader");
                    mOpenCVCameraView.enableView();
                }break;
                default:{
                    super.onManagerConnected(status);
                }break;
            }
        }
    };

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        setContentView(R.layout.activity_main);

        mOpenCVCameraView = (CameraBridgeViewBase) findViewById(R.id.cameraView);
        imageView = (ImageView) findViewById(R.id.imageView);
        predictionResults = (TextView) findViewById(R.id.classDetection) ;
        mOpenCVCameraView.setVisibility(SurfaceView.VISIBLE);
        mOpenCVCameraView.setCvCameraViewListener(cvCameraViewListener2);
        camera_index = CameraBridgeViewBase.CAMERA_ID_BACK;

        try {
            initClassifier();
            Log.i("LOAD MODEL", "Load Model Successfuly !");
        } catch (IOException e) {
            Log.d("LOAD MODEL", "Can't Load Model !");
            e.printStackTrace();
        }
    }

    @Override
    protected List<?extends CameraBridgeViewBase> getCameraViewList(){
        return Collections.singletonList(mOpenCVCameraView);
    }

    private CameraBridgeViewBase.CvCameraViewListener2 cvCameraViewListener2 = new CameraBridgeViewBase.CvCameraViewListener2() {
        @Override
        public void onCameraViewStarted(int width, int height) {
            imgBitmap = Bitmap.createBitmap(width, height, Bitmap.Config.RGB_565);
        }

        @Override
        public void onCameraViewStopped() {

        }

        @Override
        public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
            Mat rgb_mat = new Mat();
            Imgproc.cvtColor(inputFrame.rgba(), rgb_mat, Imgproc.COLOR_RGBA2RGB);
            Utils.matToBitmap(rgb_mat, imgBitmap);
           predictionResults.setText("Predict = " + recognition.predict(imgBitmap));
            return inputFrame.rgba();
        }
    };

    private void initClassifier() throws IOException {
        recognition = new Recognition("model.tflite","labels.txt",this , 224);
        recognition.init();
    }

//    public void openImage(View v) {
//        setCamera(false);
//        Intent myIntent = new Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
//        startActivityForResult(myIntent, CAMERA_REQUEST);
//    }

    public void switchCamera(){
        mOpenCVCameraView.disableView();
        if (camera_index == CameraBridgeViewBase.CAMERA_ID_FRONT){
            camera_index = CameraBridgeViewBase.CAMERA_ID_BACK;
        }
        else
            camera_index = CameraBridgeViewBase.CAMERA_ID_FRONT;
        mOpenCVCameraView.setCameraIndex(camera_index);
        mOpenCVCameraView.enableView();
    }

    public void openCamera(View v){
        setCamera(true);
        switchCamera();
    }

    private void setCamera(boolean camera_status){
        if (camera_status){
            imageView.setVisibility(View.GONE);
            imageView.clearFocus();
            mOpenCVCameraView.setVisibility(View.VISIBLE);
        }
        else
        {
            mOpenCVCameraView.clearFocus();
            mOpenCVCameraView.setVisibility(View.GONE);
            imageView.setVisibility(View.VISIBLE);
        }
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode == CAMERA_REQUEST && resultCode == RESULT_OK && data != null){
            imgUri = data.getData();
            try {
                imgBitmap = MediaStore.Images.Media.getBitmap(this.getContentResolver(), imgUri);
            }catch (IOException ex){
                ex.printStackTrace();
            }
        }
        imageView.setImageURI(imgUri);
    }

    @Override
    protected void onPause() {
        super.onPause();
        if (mOpenCVCameraView != null){
            mOpenCVCameraView.disableView();
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (mOpenCVCameraView != null){
            mOpenCVCameraView.disableView();
        }
    }

    @Override
    protected void onResume() {
        super.onResume();
        if (OpenCVLoader.initDebug()){
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
        else{
            Log.d(LOGTAG, "OpenCV Not Found !, Initializing !");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION, this, mLoaderCallback);
        }
    }
}