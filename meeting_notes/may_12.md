Layout of the lit review:
* 1: Project motivation -- Anshuk
* 1-2: Trajectory stuff -- Kris
* 4: For camera properties, extrinsics, converting image features to real-world -- Faress
* 3: Visual odometry -- Yuji
* 2-3: Feature detection -- Kris
* 1-2: Sensor fusion and Kalman filtering - Yuji


Other things (lower priority):
* Writing Introduction and motivation -- Anshuk
* Setting up Blender environment -- Anshuk
* Defining the state-space of our problem -- Kris
* Generating the trajectory -- Kris
* Make a flow chart -- Anshuk



Technical methods:
* We'll need to detect the feature in some fashion in our image.
    * April tag (?)
    * We are not learning the 3D rigid structure of the object because we already know that.
* We have an existing map of the environment.
* Structure from motion is a different problem. That's not relevant because we have an existing map.
* We can assume that our position estimate is more known than the  orientation estimate.