# Mandoct

🔬 Mandoct with CMake toolchain enabled cross-platform compilation.

- Mandoct use [gitee.com/njjixu/mangoct](https://gitee.com/njjixu/mangoct) as upstream, it develops from Mangoct-1.2.



## What are reinforced from Mangoct?

- Now can use one single 'pMatrix.jsonc' file to add offset in projection, and use the same file to do geometrical calibration.

- Now can simply add the following attribute in '*config_mgfbp.jsonc*' to do **Truncated Artifact Correction**.

  ```
  "TruncatedArtifactCorrection": true,
  ```

- TODO

## Coordinate System

<img src=".assets/coordinate system.png" alt="coordinate system" style="width:100%;" />

- The positive direction of the rotation is counterclockwise. The positive direction of the u-axis is by rotating the vector connecting the origin and the source 90 degrees clockwise.
- The object's origin is in the center of the XYZ coordinate system, while the detector's origin is on the negative X-axis. Which means detector offset must be considered when setting **pMatrix**.

## How to set pMatrix ?

- First, learn **pMatrix** definition in the projection and reconstruction procedure:
  - [projection](./PDF/pmatrix_fpj.pdf) [mgfpj]
  - [reconstruction](./PDF/pmatrix_fbp.pdf) [mgfbp]

- Then, a **pMatrix** without any offset can be obtained by the following steps:

<img src=".assets/set pMatrix.png" alt="image-20220609231113951" style="width:50%;" />

- Finally, the offset of geometric parameters can be embedded into pMatrix by some simple derivations.

>**Note:** 
>
>When use **pMatrix** function in mgfpj, "ImageRotation" should be set to 0 in mgfbp config file.
