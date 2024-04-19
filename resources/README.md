# Resources

This directory contains the many different types of files that support in the running of the LLRS.

The images subdirectory stores png files. The fake-image.png is a black photo that is used in runtime benchmarking, in lieu of the photo taken from the camera which may not be aligned properly during benchmarking. The fake-image.png is instead passed to the processor to time image processing, and then a pre-solved problem is used for timing the remaining runtime of the LLRS.
The user may store additional photos here as necessary to suit their needs. 

The problems subdirectory contains the yaml files used to define problems for use of benchmarking. When using the operational-benchmarking configuration file, you can provide any yaml file from this subdirectory as the base_problem_definition to define the problem for operational benchmarking.

The runtime subdirectory contains configuration files that are generated, used, and eventually removed by the LLRS in the execution of runtime benchmarking. This subdirectory is not to be interacted with by the user. 