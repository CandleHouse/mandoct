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
- The object's **origin** is in the center of the XYZ coordinate system, while the detector's center is on the negative X-axis with its' **origin** on one corner of the detector. Which means coordinate transformation must be considered when setting **pMatrix**.

## How to set pMatrix ?

- First, learn **pMatrix** definition in the projection and reconstruction procedure:
  - [projection](./PDF/pmatrix_fpj.pdf) [mgfpj]
  
  - [reconstruction](./PDF/pmatrix_fbp.pdf) [mgfbp]
  
- Then, due to the difference of coordinate system, **pMatrix** for this program without any offset can be obtained by taking the following steps:

  - $$
    \overrightarrow x_{do}=\overrightarrow x_{do}-DetCenterSide_u \times e_u - DetCenterSide_v \times e_v
    $$

    where:
    $$
    DetCenterSide_u=(nu-1) / 2\\
    DetCenterSide_v=(nv-1) / 2\\
    $$

  - In order to inverse the direction of u, horizontal inverse the object and inverse the rotate direction:
    $$
    \beta=-\beta\\
    pMatrix = pMatrix\cdot
    \begin{bmatrix}
    1&0&0&0\\0&-1&0&0\\0&0&1&0\\0&0&0&1
    \end{bmatrix}
    $$

- Finally, the offset of geometric parameters can be embedded into pMatrix by some simple derivations.
