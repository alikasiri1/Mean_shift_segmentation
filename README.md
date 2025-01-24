# Documentation: MeanShiftSegmentation Class

The **MeanShiftSegmentation** class implements a custom Mean Shift algorithm for image segmentation. It processes an input image by grouping pixels into clusters based on their spatial and color similarities, resulting in a segmented image. Below is the detailed documentation for each part of the implementation.

---

## Class Initialization
```python
class MeanShiftSegmentation:
    def __init__(self, spatial_radius, range_radius, max_iterations=100, use_gaussian_weights=True, handle_outliers=True):
```
### Parameters:
- **`spatial_radius`** (*float*):
  - The radius in the spatial domain (x, y coordinates) used to define the neighborhood of pixels during clustering.

- **`range_radius`** (*float*):
  - The radius in the range domain (color similarity in RGB) for clustering.

- **`max_iterations`** (*int*):
  - Maximum number of iterations for the Mean Shift algorithm.

- **`use_gaussian_weights`** (*bool*):
  - Specifies whether to apply Gaussian weights to the neighbors during clustering.

- **`handle_outliers`** (*bool*):
  - Indicates whether to handle and correct outliers in the segmented image.

---

## Preprocessing Functions
### 1. **Preprocessing the Image**
```python
def preprocess_image(self):
    image = cv2.GaussianBlur(self.input_image, (5, 5), 0)
    return image
```
- Applies Gaussian Blur to the input image to reduce noise.

### 2. **Reading the Image**
```python
def read_image(self, image_path):
    self.input_image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
```
- Reads an image from the specified path and converts it from BGR to RGB format.

---

## Segmentation Process
### Core Function: **segment**
```python
def segment(self, image):
```
Processes the image using the Mean Shift algorithm. Below are the steps:

### 1. **Converting the Image to a 5D Array**
```python
height, width, channels = image.shape
flat_image = np.zeros((height * width, 5), dtype=np.float32)

for y in range(height):
    for x in range(width):
        flat_image[y * width + x] = np.array([x, y, *image[y, x]])
```
- Converts the image into a 5D array, where each pixel is represented as `[x, y, R, G, B]`.
- **Purpose**: This format allows simultaneous processing of spatial and color data.

### 2. **Building the KD-Tree**
```python
kd_tree = KDTree(flat_image[:, :2])
```
- A KD-Tree is built using the spatial coordinates (`x, y`) of the pixels.
- **Purpose**: This structure enables efficient querying of neighboring pixels within the spatial radius.

### 3. **Querying Neighbors**
```python
spatial_neighbors = kd_tree.query_ball_point(point[:2], self.spatial_radius)
distances = np.linalg.norm(flat_image[spatial_neighbors, :2] - point[:2], axis=1)
color_distances = np.linalg.norm(flat_image[spatial_neighbors, 2:] - point[2:], axis=1)

mask = (distances < self.spatial_radius) & (color_distances < self.range_radius)
neighbors = np.array(spatial_neighbors)[mask]
```
- Queries the neighbors of a pixel within the spatial radius.
- **Purpose**: Filters neighbors based on both spatial distance (`x, y`) and color similarity (`R, G, B`).

### 4. **Applying Gaussian Weights (Optional)**
```python
if self.use_gaussian_weights:
    spatial_weights = np.exp(-(distances[mask]**2) / (2 * (self.spatial_radius**2)))
    range_weights = np.exp(-(color_distances[mask]**2) / (2 * (self.range_radius**2)))
    weights = spatial_weights * range_weights

    weighted_sum = np.sum(flat_image[neighbors] * weights[:, np.newaxis], axis=0)
    flat_image[i] = weighted_sum / np.sum(weights)
```
- Gaussian weights are calculated for neighbors based on their spatial and color distances.
- **Purpose**: Assigns higher weights to closer and more similar neighbors, ensuring smoother clustering.

### 5. **Handling Outliers (Optional)**
```python
if self.handle_outliers:
    mean_colors = np.mean(flat_image[:, 2:], axis=0)
    std_colors = np.std(flat_image[:, 2:], axis=0)
    threshold = 3 * std_colors

    for i in range(flat_image.shape[0]):
        color_distance = np.abs(flat_image[i, 2:] - mean_colors)
        if np.any(color_distance > threshold):
            flat_image[i, 2:] = mean_colors
```
- Identifies outliers as pixels whose color values deviate by more than 3 standard deviations from the mean color.
- **Purpose**: Corrects outliers by replacing them with the mean color to ensure consistency.

### 6. **Reshaping to Original Image Shape**
```python
segmented_image = flat_image[:, 2:].reshape((height, width, channels)).astype(np.uint8)
return segmented_image
```
- Converts the processed 5D array back into a 3D RGB image.
- **Purpose**: Produces the final segmented image.

---

## Visualization and Output
### 1. **Displaying Results**
```python
def display_results(self, original_image, segmented_image, process='Segmented Image'):
```
- Displays the original and segmented images side by side using Matplotlib.
- **Purpose**: Visual comparison of the original and processed images.

### 2. **Saving the Segmented Image**
```python
def save_image(self, image, output_path):
    cv2.imwrite(output_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
```
- Saves the segmented image to the specified path in BGR format.

---

## Example Usage
```python
# Initialize the MeanShiftSegmentation class
mean_shift = MeanShiftSegmentation(spatial_radius=20, range_radius=30, max_iterations=50)

# Read and preprocess the image
mean_shift.read_image('input.jpg')
image = mean_shift.preprocess_image()

# Perform segmentation
segmented_image = mean_shift.segment(image)

# Display the results
mean_shift.display_results(mean_shift.input_image, segmented_image)

# Save the segmented image
mean_shift.save_image(segmented_image, 'output.jpg')
```
---
## results
![Alt text](https://github.com/alikasiri1/Mean_shift_segmentation/blob/main/segmented_image_1.png?raw=true) 
 
![Alt text](https://github.com/alikasiri1/Mean_shift_segmentation/blob/main/segmented_image_2.png?raw=true) 

![Alt text](https://github.com/alikasiri1/Mean_shift_segmentation/blob/main/segmented_image_3.png?raw=true) 
---

## Summary
The **MeanShiftSegmentation** class provides a robust implementation of the Mean Shift algorithm, allowing for customizable spatial and color clustering. Key features include:
- Gaussian weighting for smoother segmentation.
- Outlier handling for consistent results.
- Efficient neighbor querying using a KD-Tree.

This implementation is ideal for tasks requiring image segmentation or color clustering.

