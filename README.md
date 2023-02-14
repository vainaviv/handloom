# TUSK: Learning to Trace and Untangling Semi-planar Knots
TUSK is a perception pipeline composed of the following components:
1. A learning-based iterative tracer
2. A cable state estimator using the tracer and a crossing classifier with a crossing correction algorithm. 
3. An analytic knot detection algorithm and untangling point selection aglorihtm given the cable state estimate.

## Datasets 
Datasets are located inside the ``data`` folder and organized in two categories: ``real_data`` and ``sim_data``. All data comes with an image and the necessary fields for generating ground truth labels. 

``real_data/real_data_for_tracer_viz`` shows examples of real images used in training and test time. 

The ``sim_data`` folders have been clipped to only show a few hundred examples of the train and test points. When training, each training set had ~8,000 examples and each test set had ~200 examples. More data can be generated through Blender. Details on this will be released soon!

## Tracer Model Training 
In ``config.py``, you will find the configuration for the tracer model which we found to work best: ``TRCR32_CL3_12_UNet34_B64_OS_MedleyFix_MoreReal_Sharp``. However, you are able to make your custom configurations by extending the ``BaseConfig`` class and editting hyperparameters accordingly.

To train with the current configuration, run the following command:
<pre><code>python train.py --expt_class TRCR32_CL3_12_UNet34_B64_OS_MedleyFix_MoreReal_Sharp
</code></pre>

To analyze results on crop of sim images, run: 
<pre><code>python analysis.py --checkpoint_path ../model/tracer --checkpoint_file_name tracer_model.pth
</code></pre>

To analyze on full sim images, run: 
<pre><code>python analysis.py --checkpoint_path ../model/tracer --checkpoint_file_name tracer_model.pth --trace_full_cable
</code></pre>

To analyze on full real images, run: 
<pre><code>python analysis.py --checkpoint_path ../model/tracer --checkpoint_file_name tracer_model.pth --trace_full_cable --real_world_trace
</code></pre>

If you train your own custom mode, replae the ``checkpoint_path`` to point to the directory where all checkpoints are saved. ``analysis.py`` will automatically choose the path that had the best score on the validation set. 

## Knot Detection Model Training 

## Evaluating Tracer 

## Evaluating TUSK
