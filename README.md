# CrackOnPhone

This project aims at compressing a Unet model to segment cracks on Android mobile phone. All images for training and testing are from [CFD](https://github.com/cuilimeng/CrackForest-dataset).

It includes two parts. In the "top" directory, the segmentation demo for Android is provided, which is modified from Tensorflow Lite Android demo. In the "build and compress" directory, codes for building a Unet model and compression are provided. The main techniques for compression are distilling and channels pruning (see [Reference](#reference)).

## Requirement
The *build and compress* part of the project are developed on python 3.6 and requires the following dependencies:
- tensorflow (1.11.0)
- numpy (1.14.5)
- pillow (5.2.0)
- scikit-learn (0.20.0)
- matplotlib (2.2.2)

The *top* part is modified from the tflite demo. See the page of [*top*](https://github.com/ferriswym/CrackOnPhone/tree/master/top) for details. 

Please open an issue if you meet a problem.

## Usage

### Build Unet and compress it for application
To train a unet model for inference, use the command:
```
#in /CrackOnPhone/build_and_compress/src

python unet.py  
```

You can see the generated *"model.ckpt-k"* file in the */CrackOnPhone/build_and_compress/models* directory, where k is the setting training epochs. 

After training the unet model, use:
```
#in /CrackOnPhone/build_and_compress/src

python distilling.py
```
to train a student model from the pre-trained unet model by distillation. *distilled_model.ckpt* is generated in */CrackOnPhone/build_and_compress/models* directory.

Alternatively, pruning can be used to decrease some channels in the middle layers. Use:
```
#in /CrackOnPhone/build_and_compress/src

python channel_pruning.py
```
to generate a pruned model *pruned_model.ckpt* in */CrackOnPhone/build_and_compress/models* directory.

The *evaluation.py* file in *CrackOnPhone/build_and_compress/src/tools*  is used to evaluate the models. By default each of the three scripts above  generate the *frozen.pb* in */CrackOnPhone/build_and_compress/models*(by covering) . You can run *evaluation.py* after each model generated.

You can use the scripts for custom development with modifying the data directory in *main*. Please open an issue if you meet a problem.

### Used in Android mobile phone
The *export_pb_tflite.py* file can be used to transfer *.pb* to *. tflite*, *.ckpt* to *.pb* or *.ckpt* to *.tflite*. Use it to generate a *.tflite* file, while is supported by Tensorflow Lite to run on mobile devices.

The */CrackOnPhone/top* can be imported as an Android Studio project directly. Clean and rebuild the project, replace *final.tflite* with your own *.tflite* file in *CrackOnPhone/top/app/src/main/assets*, and modify the code which calling the model in line 42 in *CrackOnPhone/top/app/src/main/java/com/example/android/tflitecamerademo/ImageSegmentationFloatUnet.java*. Then generate the *.apk* file as a common Android Studio project.
See the Tensotflow Lite Android [example](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/examples/android/app) for more details.

Please open an issue if you meet a problem.

## Reference
* [He et al., 2017] Yihui He, Xiangyu Zhang, and Jian Sun. *Channel Pruning for Accelerating Very Deep Neural Networks*. In IEEE International Conference on Computer Vision (ICCV), pages 1389-1397, 2017.
* [Hinton et al., 2015] Geoffrey Hinton, Oriol Vinyals, and Jeff Dean. *Distilling the Knowledge in a Neural Network*. CoRR, abs/1503.02531, 2015.

