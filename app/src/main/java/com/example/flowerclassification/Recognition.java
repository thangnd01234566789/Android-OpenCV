package com.example.flowerclassification;

import android.app.Activity;
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.util.Log;

import androidx.annotation.NonNull;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.sql.Array;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.stream.Collectors;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.schema.CallOptions;
import org.tensorflow.lite.schema.Uint8Vector;
import org.tensorflow.lite.support.common.FileUtil;
import org.tensorflow.lite.support.common.TensorProcessor;
import org.tensorflow.lite.support.common.ops.NormalizeOp;
import org.tensorflow.lite.support.label.TensorLabel;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

public class Recognition {
    private String modelName;
    private String labelPath;
    protected Interpreter tflite_Interpreter;
    private List<String> labelsList;
    private Activity activity;
    private int inputSize;
    private static final float IMAGE_STD = 128;
    private static final float IMAGE_MEAN = 128f;

    public Recognition(String modelName,String labelPath, Activity activity, int inputSize){
        this.modelName = modelName;
        this.activity = activity;
        this.inputSize = inputSize;
        this.labelPath = labelPath;
    }

    private MappedByteBuffer loadModel(Activity activity) throws IOException {
        AssetFileDescriptor fileDescriptor = activity.getAssets().openFd(this.modelName);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    private void loadLabel(Activity activity)throws IOException{
        labelsList = new ArrayList<String>();
        BufferedReader reader = new BufferedReader((new InputStreamReader((activity.getAssets().open(this.labelPath)))));
        String line;
        while ((line = reader.readLine()) != null){
            labelsList.add(line);
        }
        reader.close();
    }

    public void init() throws IOException{
        tflite_Interpreter = new Interpreter(loadModel(activity));
        loadLabel(activity);
    }

    private ByteBuffer convertBitmapToBuffer(Bitmap bitmap){

//        Resize Image To InputSize
        bitmap = Bitmap.createScaledBitmap(bitmap, this.inputSize, inputSize, false);
        ByteBuffer byteBuffer = ByteBuffer.allocateDirect(1 * inputSize * inputSize * 3);
        byteBuffer.order(ByteOrder.nativeOrder());
        int[] initValues = new int[inputSize * inputSize];

//        Get Color RBG Of Bitmap
        bitmap.getPixels(initValues, 0, bitmap.getWidth(), 0 , 0 , bitmap.getWidth(), bitmap.getHeight());
        int pixel = 0;
        for (int i = 0; i < inputSize; i = i + 1){
            for (int b = 0 ; b < inputSize ; b = b + 1){
                int input = initValues[pixel++];
                byteBuffer.put((byte)((input>>16) & 0xFF));
                byteBuffer.put((byte)((input>>8) & 0xFF));
                byteBuffer.put((byte)(input & 0xFF));
            }
        }
        return byteBuffer;
    }

    public String predict(Bitmap bitmap){
        ByteBuffer imageBuffer = convertBitmapToBuffer(bitmap);
        TensorBuffer probabilityBuffer = TensorBuffer.createFixedSize(new int[]{1, 5}, DataType.UINT8);
        tflite_Interpreter.run(imageBuffer, probabilityBuffer.getBuffer());
        Map<String, Float> labelProbability = convertLabel(probabilityBuffer);
        if (labelProbability != null){
            List<Float> probabilityResult = new ArrayList<Float>(labelProbability.values());
            return labelsList.get(probabilityResult.indexOf(Collections.max(probabilityResult))) + String.format(" Acc = %.2f",Collections.max(probabilityResult));
        }
        return "None";
    }

    private Map<String, Float> convertLabel(TensorBuffer probabilityBuffer){
        TensorProcessor probabilityProcesser = new TensorProcessor.Builder().add(new NormalizeOp(0, 255)).build();
        if (labelsList != null){
            TensorLabel labels = new TensorLabel(labelsList, probabilityProcesser.process(probabilityBuffer));
            Map<String, Float> floatMap = labels.getMapWithFloatValue();
            return floatMap;
        }
        return null;
    }
}
