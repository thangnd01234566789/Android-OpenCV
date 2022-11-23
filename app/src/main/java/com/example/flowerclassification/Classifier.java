package com.example.flowerclassification;

import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;

import org.tensorflow.lite.Interpreter;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.List;


public class Classifier {
    private AssetManager assetManager;
    private String modelPath;
    private String labelPath;
    private List<String> labelList;
    private int inputSize = 32;
    private static final float IMAGE_STD = 127.5f;
    private static final float IMAGE_MEAN = 127.5f;
    private Interpreter interpreter;

    public Classifier(AssetManager assetManager, String modelPath, String labelPath, int inputSize){
        this.assetManager = assetManager;
        this.modelPath = modelPath;
        this.labelPath = labelPath;
        this.inputSize = inputSize;
    }

    class Recognition{
        private String id = "";
        private String title = "";
        private float confidence = 0f;

        public Recognition(String id, String title, float confidence){
            this.id = id;
            this.title = title;
            this.confidence = confidence;
        }

        @Override
        public String toString() {
            return "Predict = {" +
                    ", title='" + title + '\'' +
                    ", confidence=" + confidence +
                    '}';
        }
    }

    private List<String> loadLabellFile(AssetManager assetManager , String labelPath) throws IOException{
        List<String> labelList = new ArrayList<>();
        BufferedReader reader = new BufferedReader(new InputStreamReader(assetManager.open(labelPath)));
        String line ;
        while ((line = reader.readLine()) != null){
            labelList.add(line);
        }
        reader.close();
        return labelList;
    }
    private static MappedByteBuffer loadModelFile(AssetManager assetManager, String modelPath){
        AssetFileDescriptor fileDescriptor = null;
        try {
            AssetFileDescriptor assetFileDescriptor = fileDescriptor = assetManager.openFd(modelPath);
            FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
            FileChannel fileChannel = inputStream.getChannel();
            long startOffest = fileDescriptor.getStartOffset();
            long declaredLength = fileDescriptor.getDeclaredLength();
            return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffest, declaredLength);
        }catch (IOException ex){
            ex.printStackTrace();
        }
        return null;
    }

    public void init() throws IOException{
        Interpreter.Options options = new Interpreter.Options();
        options.setNumThreads(5);
        options.setUseNNAPI(true);

//        Load Model
        interpreter = new Interpreter(loadModelFile(assetManager,modelPath), options);
        labelList = loadLabellFile(assetManager, labelPath);
    }

    private ByteBuffer convertBitmapToBuffer(Bitmap bitmap){
//        Resize image to 32x32
        bitmap = Bitmap.createScaledBitmap(bitmap, this.inputSize, this.inputSize, false);
        ByteBuffer byteBuffer = ByteBuffer.allocateDirect(4 * this.inputSize * this.inputSize * 3);
        byteBuffer.order(ByteOrder.nativeOrder());
        int[] intValues = new int[this.inputSize * this.inputSize];

//        Get Color RBG Of Bitmap
        bitmap.getPixels(intValues, 0 , bitmap.getWidth(), 0 ,0 ,bitmap.getWidth(), bitmap.getHeight());
        int pixel = 0 ;
        for (int i = 0 ; i < this.inputSize; i = i + 1){
            for (int b = 0 ; b < this.inputSize; b = b + 1){
                int input = intValues[pixel++];

                byteBuffer.putFloat((((input>>16 & 0xFF) - IMAGE_MEAN) / IMAGE_STD));
                byteBuffer.putFloat((((input>>8 & 0xFF) - IMAGE_MEAN) / IMAGE_STD));
                byteBuffer.putFloat((((input & 0xFF) - IMAGE_MEAN) / IMAGE_STD));
            }
        }
        return byteBuffer;
    }
}
