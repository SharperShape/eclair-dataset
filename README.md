<a href="https://sharpershape.com/our-solutions/living-digital-twin/" target="_blank">
  <img src="assets/cropped-SS-logo.png" width="240">
</a>

#### Research @ Sharper Shape (CVPRW 2024)

# ECLAIR: A High-Fidelity Aerial LiDAR Dataset for Semantic Segmentation

This is the official repository of the **ECLAIR** dataset. For technical details, please refer to:

**ECLAIR: A High-Fidelity Aerial LiDAR Dataset for Semantic Segmentation** <br />
[Iaroslav Melekhov](https://imelekhov.com/), [Anand Umashankar](https://www.linkedin.com/in/anandcu3/), [Hyeong-Jin Kim](https://www.linkedin.com/in/hjedkim/), 
[Vlad Serkov](https://www.linkedin.com/in/vladserkoff/), [Dusty Argyle](https://www.linkedin.com/in/dustinargyle/). <br />
**[[Paper](https://arxiv.org/abs/2404.10699)] [[Project page]] [[Download]]
[[USM workshop@CVPR2024](https://usm3d.github.io/)]** <br />

We introduce ECLAIR (Extended Classification of Lidar for AI Recognition), a new outdoor large-scale aerial LiDAR dataset designed specifically for advancing research in point cloud semantic segmentation. As the most extensive and diverse collection of its kind to date, the dataset covers a total area of 10km^2 with close to 600 million points and features eleven distinct object categories. To guarantee the dataset’s quality and utility, we have thoroughly curated the point labels through an internal team of experts, ensuring accuracy and consistency in semantic labeling

<p align="center"> <img src="assets/teaser.png" width="100%"> </p>

Semantic annotations (classes):
- **Ground**: all points representing the Earth’s surface, including, soil, pavement, roads, and the bottom of water bodies.
- **Vegetation**: all points representing organic plant life, ranging from trees, low shrubs, and tall grass of all heights.
- **Buildings**: man-made structures characterized by roofs and walls, encompassing houses, factories, and sheds
- **Transmission Wires**: high-voltage wires for longdistance transmission from power plants to substations. Either directly connected to transmission towers or poles. Also includes transmission ground wires.
- **Distribution Wires**: Lower-voltage overhead distribution wires distributing electricity from substation to end
users. Includes span guy wires and communication wires.
- **Poles**: Utility poles used to support different types of wires or electroliers. These can include poles with either transmission or distribution wires. Down guy wires, crossarms and transformers are also included in this class.
- **Transmission Towers**: Large structures supporting transmission wires with the distinct characterisation of steel lattices and cross beams.
- **Fence**:  Barriers, railing, or other upright structure, typically of wood or wire, enclosing an area of ground.
- **Vehicle**: All wheeled vehicles that can be driven.
- **Unassigned**: This category serves as a catch-all for nonsubject points. Anything that is not on the class list is classified as Unassigned. These include wooden pallets, trash, structures not large or strong enough to put under buildings (tents, boulders, etc.), and house antennas.

### Runing the demo
This demo will run a pre-trained model on the test split of the proposed dataset and compute the macro and per-class F1 score. You will be asked to provide a path to the model checkpoint (referred as `$MODEL_WEIGHTS`) and a dataset directory (referred as `$DATA_DIR`). Create them wherever you wish and make sure to provide their absolute paths.

* Download the model weights (checkpoint) using [this link](https://cdn.shared.sharpershape.services/eclair-dataset/model_eclair.pth) and place it under `$MODEL_WEIGHTS/model_eclair.pth`
* Download the ECLAIR dataset by filling in the following form. The dataset is distributed under [Creative Commons 4.0 (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode.en). Once downloaded and unzipped, one should get the following structure:
```
$DATA_DIR
|-- pointclouds
|       |-- pointcloud_1.laz
|       |-- pointcloud_2.laz
|       `-- ...
|-- labels.json
```
* Run `test.py` as follows:
  ```
  python test.py --dataset_path $DATA_DIR --model_weights_file $MODEL_WEIGHTS/model_eclair.pth
  ```

### Copyright
The annotated dataset is licensed under [Creative Commons 4.0 (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode.en). This data may not be used for commercial purposes. We will consider commercial usage on a case-by-case basis. Please contact us for more information.

### Contact
For additional questions about the dataset or inquiries about commercial use, please contact <anand.umashankar@sharpershape.com>

### Citation
If you find our work useful, please consider citing:

    @inproceedings{eclair2024,
        title={ECLAIR: A High-Fidelity Aerial LiDAR Dataset for Semantic Segmentation},
        author={Melekhov, Iaroslav and Umashankar, Anand and Kim, Hyeong-Jin and Serkov, Vladislav and Argyle, Dusty},
        booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
        year={2024}
    }

