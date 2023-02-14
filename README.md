# TUSK: Learning to Trace and Untangling Semi-planar Knots
TUSK is a perception pipeline composed of the following components:
1. A learning-based iterative tracer
2. A cable state estimator using the tracer and a crossing classifier with a crossing correction algorithm. 
3. An analytic knot detection algorithm and untangling point selection aglorihtm given the cable state estimate.

## Datasets 
Datasets are located inside the ``data`` folder and organized in two categories: ``real_data`` and ``sim_data``. The ``sim_data`` folders have been clipped to only show a few hundred examples of the train and test points. When training, each training set had ~8,000 examples and each test set had ~200 examples. More data can be generated through Blender. Details on this will be released soon!

## Tracer Model Training 

## Knot Detection Model Training 

## Evaluating Tracer 

## Evaluating TUSK
