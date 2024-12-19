<h2>Tensorflow-Image-Segmentation-IHC (2024/12/20)</h2>

This is the first experiment of Image Segmentation for ImmunoHistoChemistry(IHC) 
 based on 
the latest <a href="https://github.com/sarah-antillia/Tensorflow-Image-Segmentation-API">Tensorflow-Image-Segmentation-API</a>, 
and  <a href="https://drive.google.com/file/d/1Q8V4x9XMq8kWAeBEE1svS2NbfrCy40It/view?usp=sharing">
IHC-ImageMask-Dataset.zip</a>, which was derived by us from 
<a href="https://warwick.ac.uk/fac/sci/dcs/research/tia/data/nuclick/ihc_nuclick.zip">
https://warwick.ac.uk/fac/sci/dcs/research/tia/data/nuclick/ihc_nuclick.zip
</a>, which is NuClick IHC annotations for crops from the LYON19 dataset,<br>
<br>
<hr>
<b>Actual Image Segmentation for Images of 512x512 pixels</b><br>
As shown below, the inferred masks look similar to the ground truth masks. <br>

<table>
<tr>
<th>Input: image</th>
<th>Mask (ground_truth)</th>
<th>Prediction: inferred_mask</th>
</tr>
<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/IHC/mini_test/images/ROI_344_2.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/IHC/mini_test/masks/ROI_344_2.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/IHC/mini_test_output/ROI_344_2.jpg" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/IHC/mini_test/images/ROI_343_1.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/IHC/mini_test/masks/ROI_343_1.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/IHC/mini_test_output/ROI_343_1.jpg" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/IHC/mini_test/images/ROI_348_1.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/IHC/mini_test/masks/ROI_348_1.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/IHC/mini_test_output/ROI_348_1.jpg" width="320" height="auto"></td>
</tr>
</table>

<hr>
<br>
In this experiment, we used the simple UNet Model 
<a href="./src/TensorflowUNet.py">TensorflowSlightlyFlexibleUNet</a> for this IHCSegmentation Model.<br>
As shown in <a href="https://github.com/sarah-antillia/Tensorflow-Image-Segmentation-API">Tensorflow-Image-Segmentation-API</a>.
you may try other Tensorflow UNet Models:<br>

<li><a href="./src/TensorflowSwinUNet.py">TensorflowSwinUNet.py</a></li>
<li><a href="./src/TensorflowMultiResUNet.py">TensorflowMultiResUNet.py</a></li>
<li><a href="./src/TensorflowAttentionUNet.py">TensorflowAttentionUNet.py</a></li>
<li><a href="./src/TensorflowEfficientUNet.py">TensorflowEfficientUNet.py</a></li>
<li><a href="./src/TensorflowUNet3Plus.py">TensorflowUNet3Plus.py</a></li>
<li><a href="./src/TensorflowDeepLabV3Plus.py">TensorflowDeepLabV3Plus.py</a></li>

<br>

<h3>1. Dataset Citation</h3>
The dataset used here has been take from the following web site<br>
<br>
<a href="https://warwick.ac.uk/fac/sci/dcs/research/tia/data/nuclick/ihc_nuclick.zip">
https://warwick.ac.uk/fac/sci/dcs/research/tia/data/nuclick/ihc_nuclick.zip
</a>, which is NuClick IHC annotations for crops from the LYON19 dataset,<br>
<br>
On NuClick, please refer to 
<a href="https://wrap.warwick.ac.uk/id/eprint/139218/1/WRAP-nuclick-Rajpoot-2020.pdf">
https://wrap.warwick.ac.uk/id/eprint/139218/1/WRAP-nuclick-Rajpoot-2020.pdf
</a>

<br>
<h3>
<a id="2">
2 IHC ImageMask Dataset
</a>
</h3>
 If you would like to train this IHC Segmentation model by yourself,
 please download the dataset from the google drive  
<a href="https://drive.google.com/file/d/1Q8V4x9XMq8kWAeBEE1svS2NbfrCy40It/view?usp=sharing">
IHC-ImageMask-Dataset.zip</a>
, expand the downloaded ImageMaskDataset and put it under <b>./dataset</b> folder to be
<pre>
./dataset
└─IHC
    ├─test
    │   ├─images
    │   └─masks
    ├─train
    │   ├─images
    │   └─masks
    └─valid
        ├─images
        └─masks
</pre>
<br>
<br>
<b>IHC Statistics</b><br>
<img src ="./projects/TensorflowSlightlyFlexibleUNet/IHC/IHC_Statistics.png" width="512" height="auto"><br>
<br>
As shown above, the number of images of train and valid datasets is not enough to use for a training set of our segmentation model.
<br>
<br>
<b>Train_images_sample</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/IHC/asset/train_images_sample.png" width="1024" height="auto">
<br>
<b>Train_masks_sample</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/IHC/asset/train_masks_sample.png" width="1024" height="auto">
<br>

<h3>
3 Train TensorflowUNet Model
</h3>
 We have trained IHCTensorflowUNet Model by using the following
<a href="./projects/TensorflowSlightlyFlexibleUNet/IHC/train_eval_infer.config"> <b>train_eval_infer.config</b></a> file. <br>
Please move to ./projects/TensorflowSlightlyFlexibleUNet/IHC and run the following bat file.<br>
<pre>
>1.train.bat
</pre>
, which simply runs the following command.<br>
<pre>
>python ../../../src/TensorflowUNetTrainer.py ./train_eval_infer.config
</pre>
<hr>

<b>Model parameters</b><br>
Defined a small <b>base_filters</b> and large <b>base_kernels</b> for the first Conv Layer of Encoder Block of 
<a href="./src/TensorflowUNet.py">TensorflowUNet.py</a> 
and a large num_layers (including a bridge between Encoder and Decoder Blocks).
<pre>
[model]
base_filters   = 16
base_kernels   = (11,11)
num_layers     = 8
dilation       = (4,4)
</pre>

<b>Learning rate</b><br>
Defined a small learning rate.  
<pre>
[model]
learning_rate  = 0.0001
</pre>

<b>Online augmentation</b><br>
Disabled our online augmentation tool. 
<pre>
[model]
model         = "TensorflowUNet"
generator     = False
</pre>

<b>Loss and metrics functions</b><br>
Specified "bce_dice_loss" and "dice_coef".<br>
<pre>
[model]
loss           = "bce_dice_loss"
metrics        = ["dice_coef"]
</pre>
<b >Learning rate reducer callback</b><br>
Enabled learing_rate_reducer callback, and a small reducer_patience.
<pre> 
[train]
learning_rate_reducer = True
reducer_factor     = 0.3
reducer_patience   = 4
</pre>

<b>Early stopping callback</b><br>
Enabled early stopping callback with patience parameter.
<pre>
[train]
patience      = 10
</pre>

<b>Enabled color space converter</b><br>
<pre>
[image]
color_converter = "cv2.COLOR_BGR2HSV"
</pre>

<b>Epoch change inference callbacks</b><br>
Enabled epoch_change_infer callback.<br>
<pre>
[train]
epoch_change_infer       = True
epoch_change_infer_dir   =  "./epoch_change_infer"
epoch_changeinfer        = False
epoch_changeinfer_dir    = "./epoch_changeinfer"
num_infer_images         = 6
</pre>

By using this callback, on every epoch_change, the inference procedure can be called
 for 6 images in <b>mini_test</b> folder. This will help you confirm how the predicted mask changes 
 at each epoch during your training process.<br> <br> 

<b>Epoch_change_inference output</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/IHC/asset/epoch_change_infer.png" width="1024" height="auto"><br>
<br>

In this experiment, the training process was stopped at epoch 32  by EarlyStopping Callback.<br><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/IHC/asset/train_console_output_at_epoch_32.png" width="720" height="auto"><br>
<br>

<a href="./projects/TensorflowSlightlyFlexibleUNet/IHC/eval/train_metrics.csv">train_metrics.csv</a><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/IHC/eval/train_metrics.png" width="520" height="auto"><br>

<br>
<a href="./projects/TensorflowSlightlyFlexibleUNet/IHC/eval/train_losses.csv">train_losses.csv</a><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/IHC/eval/train_losses.png" width="520" height="auto"><br>

<br>

<h3>
4 Evaluation
</h3>
Please move to a <b>./projects/TensorflowSlightlyFlexibleUNet/IHC</b> folder,<br>
and run the following bat file to evaluate TensorflowUNet model for IHC.<br>
<pre>
./2.evaluate.bat
</pre>
This bat file simply runs the following command.
<pre>
python ../../../src/TensorflowUNetEvaluator.py ./train_eval_infer_aug.config
</pre>

Evaluation console output:<br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/IHC/asset/evaluate_console_output_at_epoch_32.png" width="720" height="auto">
<br><br>Image-Segmentation-IHC

<a href="./projects/TensorflowSlightlyFlexibleUNet/IHC/evaluation.csv">evaluation.csv</a><br>

The loss (bce_dice_loss) to this IHC/test was not low, and dice_coef not high as shown below.
<br>
<pre>
loss,0.2724
dice_coef,0.6397
</pre>
<br>

<h3>
5 Inference
</h3>
Please move to a <b>./projects/TensorflowSlightlyFlexibleUNet/IHC</b> folder<br>
,and run the following bat file to infer segmentation regions for images by the Trained-TensorflowUNet model for IHC.<br>
<pre>
./3.infer.bat
</pre>
This simply runs the following command.
<pre>
python ../../../src/TensorflowUNetInferencer.py ./train_eval_infer_aug.config
</pre>
<hr>
<b>mini_test_images</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/IHC/asset/mini_test_images.png" width="1024" height="auto"><br>
<b>mini_test_mask(ground_truth)</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/IHC/asset/mini_test_masks.png" width="1024" height="auto"><br>

<hr>
<b>Inferred test masks</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/IHC/asset/mini_test_output.png" width="1024" height="auto"><br>
<br>
<hr>
<b>Enlarged images and masks </b><br>

<table>
<tr>
<th>Image</th>
<th>Mask (ground_truth)</th>
<th>Inferred-mask</th>
</tr>

<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/IHC/mini_test/images/ROI_338_2.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/IHC/mini_test/masks/ROI_338_2.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/IHC/mini_test_output/ROI_338_2.jpg" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/IHC/mini_test/images/ROI_343_1.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/IHC/mini_test/masks/ROI_343_1.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/IHC/mini_test_output/ROI_343_1.jpg" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/IHC/mini_test/images/ROI_346_1.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/IHC/mini_test/masks/ROI_346_1.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/IHC/mini_test_output/ROI_346_1.jpg" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/IHC/mini_test/images/ROI_348_1.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/IHC/mini_test/masks/ROI_348_1.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/IHC/mini_test_output/ROI_348_1.jpg" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/IHC/mini_test/images/ROI_360_2.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/IHC/mini_test/masks/ROI_360_2.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/IHC/mini_test_output/ROI_360_2.jpg" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/IHC/mini_test/images/ROI_347_2.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/IHC/mini_test/masks/ROI_347_2.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/IHC/mini_test_output/ROI_347_2.jpg" width="320" height="auto"></td>
</tr>
</table>
<hr>
<br>


<h3>
References
</h3>
<b>1. NuClick: A Deep Learning Framework for Interactive Segmentation of Microscopic Images</b><br>
Navid Alemi Koohbanania, Mostafa Jahanifarc, Neda Zamani Tajadind, Nasir Rajpoota<br>

<a href="https://wrap.warwick.ac.uk/id/eprint/139218/1/WRAP-nuclick-Rajpoot-2020.pdf">
https://wrap.warwick.ac.uk/id/eprint/139218/1/WRAP-nuclick-Rajpoot-2020.pdf
</a>
<br>
