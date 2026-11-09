# README illustration

Generated with the built-in `image_gen` tool. The illustration explains the task;
it is not a diagram of the implemented network or a measured trajectory.

## Prompt

Use case: scientific-educational
Asset type: wide header illustration for a GitHub README, approximately 2:1 landscape.
Primary request: Explain the TartanIMU challenge at a glance: four very different moving platforms share the task of estimating motion from inertial sensors.
Composition: clean editorial scientific illustration on an off-white background. Across the left half, four recognizable subjects: a compact quadruped robot dog, a car, a walking person holding a small sensor, and a quadcopter drone. Show a small marked inertial sensor on each, connected by subtle thin lines to a central IMU signal panel containing six waveform traces. From that panel a single clean arrow leads to a neural model block, then an output panel on the right showing a 3D velocity vector and a curved trajectory with a few sample points. Keep a wide, balanced horizontal composition, generous white space, restrained blue and orange accents, clear silhouettes, precise technical magazine illustration.
Text (verbatim), only these short labels: "TartanIMU", "IMU", "200 × 6", "Model", "Velocity → trajectory".
Scientific constraints: the actual neural network predicts mean 3D body-frame velocity for each one-second IMU window. The trajectory is downstream integration, so show the velocity arrow before the trajectory; do not imply the model directly predicts absolute position or that IMU alone supplies global orientation. No numerical results or metric claims.
Avoid: photographic clutter, marketing slogans, tiny annotations, invented logos, external cameras or GPS sensors, watermarks.
