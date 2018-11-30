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
import android.graphics.Bitmap;
import java.io.IOException;

public class ImageSegmentationFloatUnet extends ImageSegmentation{

    // Unet requires additional normalization of the used input.
    private static final float IMAGE_MEAN = 127.5f;
    private static final float IMAGE_STD = 127.5f;

    // An array to hold output image, to feed into Tensorflow Lite as outputs.
    private float[][][][] outData = null;

    // Initializes an ImageClassifier.
    ImageSegmentationFloatUnet(Activity activity) throws IOException{
        super(activity);
        // TODO get the outputImage size from input.
        outData = new float[1][getImageSizeX()][getImageSizeY()][1];
    }

    @Override
    protected String getModelPath() {
        return "final.tflite";
    }

    @Override
    protected int getImageSizeX() { return 320; }

    @Override
    protected int getImageSizeY() { return 480; }

    @Override
    protected int getNumBytesPerChannel() {
        return 4;
    }

    @Override
    protected void addPixelValue(int pixelValue) {
        imgData.putFloat((((pixelValue >> 16) & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
        imgData.putFloat((((pixelValue >> 8) & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
        imgData.putFloat(((pixelValue & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
    }

    @Override
    protected void getOutputImage(Bitmap outImage) {
        for (int i = 0; i < getImageSizeY(); ++i)
            for (int j = 0; j < getImageSizeX(); ++j) {
                int color = 0xff000000;
                int tmp = (int) (outData[0][j][i][0] * 255);
                color |= tmp << 16 | tmp << 8 | tmp;
                outImage.setPixel(i, j, color);
            }
    }

    @Override
    public void runInference() { tflite.run(imgData, outData); }
}
