/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// Codes are Tensorflow Lite, modified from classification to segmentation

package com.example.android.tflitecamerademo;

import android.app.Activity;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.os.Environment;
import android.os.SystemClock;
import android.text.SpannableString;
import android.text.SpannableStringBuilder;
import android.text.style.ForegroundColorSpan;
import android.util.Log;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;

import org.tensorflow.lite.Interpreter;

// Segmentation with Tensorflow Lite.

public abstract class ImageSegmentation {
    // Dispaly preferences
    private static final float GOOD_PROB_TRRESHOLD = 0.3f;
    private static int SMALL_COLOR = 0xffddaa88;

    // Tag for the Log.
    private static final String TAG = "TfLiteCameraDemo";

    // Dimensions of inputs.
    private static final int DIM_BATCH_SIZE = 1;

    private static final int DIM_PIXEL_SIZE = 3;

    // params for save images.
    private static int saveNums = 0;

    // Preallocated buffers for storing image data in.
    private int[] intValues = new int[getImageSizeX() * getImageSizeY()];

    // Options for configuring the Interpreter.
    private final Interpreter.Options tfliteOptions = new Interpreter.Options();

    // The loaded TensorFlow Lite model.
    private MappedByteBuffer tfliteModel;

    // A bitmap to hold the output image.
    protected Bitmap outImage;

    // An instance of the driver class to run model inference with TensorFlow lite.
    protected Interpreter tflite;

    // A ByteBuffer to hold image data, to be feed into TensorFlow lite as inputs.
    protected ByteBuffer imgData = null;

    // Initializes ImageSegmentation.
    ImageSegmentation(Activity activity) throws IOException {
        tfliteModel = loadModelFile(activity);
        tflite = new Interpreter(tfliteModel, tfliteOptions);
        imgData =
                ByteBuffer.allocateDirect(
                        DIM_BATCH_SIZE
                        * getImageSizeX()
                        * getImageSizeY()
                        * DIM_PIXEL_SIZE
                        * getNumBytesPerChannel());
        imgData.order(ByteOrder.nativeOrder());
        Log.d(TAG, "Created a TensorFlow Lite Image Classifier.");
    }

    // Segment a frame from the preview stream.
    void segmentFrame(Bitmap input, Bitmap output, SpannableStringBuilder builder) {
        if (tflite == null) {
            Log.d(TAG, "Image segmentation has not been initialized; Skipped.");
            builder.append(new SpannableString("Uninitialized Segmentation."));
        }
        converBitmapToByteBuffer(input);
        // The magic part!!!
        long startTime = SystemClock.uptimeMillis();
        runInference();
        long endTime = SystemClock.uptimeMillis();
        Log.d(TAG, "Timecost to run model inference: " + Long.toString(endTime - startTime));
        getOutputImage(output);

        // Print the results.
        long duration = endTime - startTime;
        SpannableString span = new SpannableString(duration + " ms");
        span.setSpan(new ForegroundColorSpan(android.graphics.Color.LTGRAY), 0, span.length(), 0);
        builder.append(span);
        if (saveNums < 5){
            ++saveNums;
            Log.d(TAG, "SAVENUMBER: " + saveNums);
            String filename = String.valueOf(saveNums);
//            if (saveNums % 30 == 0) {
            writeBitmapToLocal(outImage, filename);
            Log.d(TAG, "susscessfully saved: " + filename);
//            }
        }
    }

    public long segmentFrame(Bitmap input, Bitmap output){
        long startTime = SystemClock.uptimeMillis();
        converBitmapToByteBuffer(input);
        long convertInputTime = SystemClock.uptimeMillis();
        Log.d(TAG, "Timecost to convert input into ByteBuffer: " + Long.toString(convertInputTime - startTime));
        runInference();
        long inferenceTime = SystemClock.uptimeMillis();
        Log.d(TAG, "Timecost to run an inference: " + Long.toString(inferenceTime - convertInputTime));
        getOutputImage(output);
        long convertOutputTime = SystemClock.uptimeMillis();
        Log.d(TAG, "Timecost to convert output into a bitmap: " + Long.toString(convertOutputTime - inferenceTime));
        return inferenceTime - convertInputTime;
    }

    private void recreateInterpreter() {
        if (tflite != null) {
            tflite.close();
            tflite = new Interpreter(tfliteModel, tfliteOptions);
        }
    }

    public void setUseNNAPI(Boolean nnapi) {
        tfliteOptions.setUseNNAPI(nnapi);
        recreateInterpreter();
    }

    public void setNumThreads(int numThreads) {
        tfliteOptions.setNumThreads(numThreads);
        recreateInterpreter();
    }

    // Closes tflite to release resources.
    public void close() {
        tflite.close();
        tflite = null;
        tfliteModel = null;
    }

    // Memory-map the model file in Assets.
    private MappedByteBuffer loadModelFile(Activity activity) throws IOException {
        AssetFileDescriptor fileDescriptor = activity.getAssets().openFd(getModelPath());
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    // Write Image data into a ByteBuffer.
    public void converBitmapToByteBuffer(Bitmap bitmap){
        if (imgData == null){
            return;
        }
        imgData.rewind();
        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
        // Convert the image to floating point.
        int pixel = 0;
        long startTime = SystemClock.uptimeMillis();
        for (int i = 0; i < getImageSizeX(); ++i) {
            for (int j = 0; j < getImageSizeY(); ++j) {
                final int val = intValues[pixel++];
                addPixelValue(val);
            }
        }
        long endTime = SystemClock.uptimeMillis();
        Log.d(TAG, "Timecost to put values into ByteBuffer: " + Long.toString(endTime - startTime));
    }

    // Write Image to local directory.
    private void writeBitmapToLocal(Bitmap bitmap, String filename) {
        // Obtain the directory.
        String dir = Environment.getExternalStorageDirectory().getAbsolutePath() + "/TensorflowLite";
        File path = new File(dir);
        if (!path.exists())
            path.mkdirs();

        // Ensure for the state of external storage.
        String state = Environment.getExternalStorageState();
        if (!state.equals(Environment.MEDIA_MOUNTED))
            return;

        // Save Bitmap to local.
        try{
            File file = new File(dir + "/" + filename + ".png");
            FileOutputStream out = new FileOutputStream(file);
            bitmap.compress(Bitmap.CompressFormat.JPEG, 100, out);
            out.flush();
            out.close();
        } catch (Exception e){
            e.printStackTrace();
        }
    }

    // Get the name of the model file stored in Assets.
    protected abstract String getModelPath();

    // Get the image size along the x axis.
    protected abstract int getImageSizeX();

    // Get the image size along the y axis.
    protected abstract int getImageSizeY();

    // Get the number of bytes that is used to store a single color channel value.
    protected abstract int getNumBytesPerChannel();

    // Add pixelValue to byteBuffer.
    protected abstract void addPixelValue(int pixelValue);

    // Run inference using the prepared input in {#imgData}.
    public abstract void runInference();

    // Get probability bitmap from output byte.
    protected abstract void getOutputImage(Bitmap outImage);
}
