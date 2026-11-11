# ROIs and masks

This page explains how regions of interest (ROIs) and masks are defined in the **Sediment** widget, how they are stored in a project, and how they are used by the other widgets (**HyperAnalysis**, **SpectralIndices**, **Concatenation** and batch plotting). It also describes what happens depending on the order in which ROIs and masks are created.

## Coordinates

All positions are given as *(row, column)* pixel coordinates, rows running along the length of the core (vertical in the viewer) and columns across it. A rectangular ROI is stored as its four corners, and is used through its *bounds*: `row_bounds = [row_min, row_max]` and `col_bounds = [col_min, col_max]`. Image regions are extracted as `image[row_min:row_max, col_min:col_max]`, so the maximum is excluded and the region has the shape `(row_max - row_min, col_max - col_min)`. ROI corners are rounded to integers when an ROI is drawn or moved.

## Types of ROIs

| ROI | Layer (Sediment widget) | Coordinates | Purpose |
|---|---|---|---|
| Main ROI | `main-roi` (blue) | full image | The part of the image that is analyzed. One project can contain several main ROIs (e.g. several sections of a core in one image). |
| Sub-ROI | `rois_0`, `rois_1`, ... (red), one layer per main ROI | full image | Narrower regions within a main ROI. The first sub-ROI of each main ROI is used by the HyperAnalysis widget, and as default measurement ROI in the SpectralIndices widget. |
| Measurement ROI | `rois` (red, SpectralIndices widget) | relative to the main ROI | Columns over which index maps are averaged to compute index projections. |

### Creating ROIs in the Sediment widget

- **When an image is selected**, a main ROI covering the full image is added automatically.
- **Add main ROI** adds a rectangle spanning all rows of the current crop (see below), centered on it, with the width set in *Main ROI width*.
- Main ROIs can be edited directly in the `main-roi` layer. **Import main ROI from file** replaces the main ROIs with the ones from another project's `Parameters.yml`.
- **Sub-ROIs** are added by double-clicking in the viewer: a rectangle of width *Sub-ROI width*, spanning all rows of the current crop, is added to the `rois_<n>` layer of the main ROI selected in *Selected main ROI*.
- If a main ROI has no sub-ROI when the project is exported, the main ROI itself is used as sub-ROI.

### Saving ROIs

ROIs are saved in `Parameters.yml` in the project folder when exporting the project (**Export Project**) and when correcting the full image (**Correct and save data**):

- `main_roi`: list of main ROIs, each as 8 numbers (4 corners).
- `rois`: for each main ROI, a list of sub-ROIs.
- `measurement_roi`: for each main ROI, the measurement ROI (saved by the SpectralIndices widget, relative to the main ROI). If empty, the first sub-ROI is used.

For each main ROI `n`, outputs are saved in a folder `roi_<n>` (mask, index maps, plots, etc.).

## Cropping in the Sediment widget

The Sediment widget always displays a *crop* of the image, defined by its current `row_bounds` and `col_bounds`:

- After selecting an image, the crop is the full image.
- **Crop with main** sets the crop to the main ROI selected in *Selected main ROI*.
- **Reset crop** sets the crop back to the full image.
- Changing *Selected main ROI* automatically crops with the newly selected main ROI (see [Masks of multiple main ROIs](multiple-rois)).
- **Import Project** loads the ROIs and crops with the first main ROI.

Only the data of the crop are loaded. The image layers (`imcube`, `red`, `green`, `blue`, ...) and mask layers have the size of the crop and are *translated* in the viewer to their position in the full image, so that the viewer coordinates always correspond to full image coordinates, whatever the crop.

```{note}
The full image is always corrected and saved by **Correct and save data**, independently of the crop. The ROIs only define which part of `corrected.zarr` is used in the subsequent analysis.
```

## Masks

A mask marks pixels to be **excluded** from the analysis (value 1), e.g. cracks, holes or borders. Masks are created in the **Mask** tab of the Sediment widget:

| Mask layer | Created by |
|---|---|
| `manual-mask` | **Drawing**: empty layer to paint on |
| `border-mask` | **Border**: automatic detection of the borders of the core |
| `intensity-mask` | **Man. thresh.** / **Auto thresh.**: pixels outside the intensity range |
| `ml-mask` | **ML**: pixels classified as class 1 by the pixel classifier |
| `complete-mask` | **Combine masks**: union of all the masks above |
| `clean-mask` | **Clean mask**: cleaned version of the complete mask |

All masks are created with the size of the **current crop** and are placed at its position in the viewer. When the crop changes (**Crop with main**, **Reset crop**, change of *Selected main ROI*), existing masks are kept and adjusted to the new crop. A full image copy of each mask is kept in memory: the region of the current crop is updated in this copy, and the new crop is extracted from it. Masks thus keep their position in the image and nothing is lost when cropping: e.g. a mask created on the full image is restored by **Reset crop** after **Crop with main**. Regions that were never covered by a mask are not masked (0). All masks always have the size of the current crop and can be combined. The same applies to the `annotations` and `segmentation` layers of the pixel classifier. The copy of a mask is discarded when its layer is deleted (and when another image is opened).

Note that editing a mask only changes the region of the current crop. For example, recomputing the `intensity-mask` while cropped with a main ROI keeps the previous intensity mask outside that main ROI.

### Saving and loading masks

**Save mask** (and **Export Project**) saves the mask to `roi_<n>/mask.tif`, for the main ROI selected in *Selected main ROI* or for all main ROIs, depending on the mask mode (see [Masks of multiple main ROIs](multiple-rois)). The saved mask is `clean-mask` if it exists, otherwise `complete-mask`, otherwise the combination of the existing masks.

The saved mask **always covers exactly the selected main ROI**: whatever crop the mask was created on, the region corresponding to the main ROI is extracted using the mask position in the viewer. Parts of the mask outside the main ROI are discarded, and parts of the main ROI not covered by the mask are saved as not masked (0).

When loading a mask (**Load mask**, **Import Project**, **Crop with main** if no mask layer is present, change of *Selected main ROI* with separate masks), `mask.tif` is loaded as `complete-mask`, placed at the position of the main ROI and cropped or extended to the current crop.

(multiple-rois)=
### Masks of multiple main ROIs

Each main ROI has its own mask file, `roi_<n>/mask.tif`, used by the other widgets. In the Sediment widget, the **Masks** option of the ROI tab defines how masks are handled when there are several main ROIs:

**Separate per main ROI** (default): each main ROI has its own masks, and the Sediment widget displays the masks of a single main ROI at a time. When *Selected main ROI* is changed:

1. the current masks are **automatically saved** to the `mask.tif` of the previously selected main ROI (a project folder is needed; if none is set you are asked to select one, otherwise the masks are discarded with a warning),
2. all mask layers are removed,
3. the image is cropped with the newly selected main ROI,
4. the saved `mask.tif` of the newly selected main ROI is loaded, if it exists.

Switching back and forth between main ROIs thus keeps the masks of each main ROI. Note that the automatic saving overwrites the existing `mask.tif` of the previous main ROI. **Save mask** and **Export Project** save the mask of the selected main ROI.

**Shared by all main ROIs**: one set of masks covers the whole image and applies to all main ROIs, e.g. a mask created on the full image is used for all main ROIs. When *Selected main ROI* is changed, the masks are only cropped with the newly selected main ROI: nothing is saved, removed or loaded.

- **Combine masks** combines the masks over the full image.
- **Clean mask** keeps the largest unmasked region and fills its holes. As this is done for a single sediment section, it is applied to the current crop for display and, when saving, to each main ROI separately.
- **Save mask** and **Export Project** save the mask of **all** main ROIs, each cropped from the shared mask (and cleaned if a `clean-mask` exists).
- **Load mask** and **Import Project** rebuild the shared mask from the saved masks of all main ROIs.

The mode can be changed at any time. It only affects what happens when the selected main ROI changes and when saving or loading.

```{note}
Older versions of the plugin saved masks with the extent of the crop active when saving. If a mask was saved without cropping, the file covers the full image. Such masks are detected when loading (the file is larger than the main ROI) and the main ROI region is extracted, so that they are displayed at the correct position in all widgets.
```

### Order of operations

**Recommended order**: correct the image (**Correct and save data**), define the main ROI(s), then:

- with **separate** masks: **Crop with main**, create the masks, **Save mask**/**Export Project**. For further main ROIs, change *Selected main ROI* (this saves the masks of the previous main ROI and crops with the new one), create the masks and save.
- with **shared** masks: create the masks on the full image or on any crop, check them by changing *Selected main ROI*, then **Save mask**/**Export Project** to save the mask of all main ROIs.

The table below describes what happens in other cases.

| Situation | What happens |
|---|---|
| Masks created on the full image (before **Crop with main** or after **Reset crop**), then saved | The main ROI region is extracted and saved. Masked pixels outside the main ROI are discarded. |
| Masks created on the full image, then **Crop with main** | The masks are kept and cropped to the main ROI. The regions outside the main ROI are kept in memory and restored by **Reset crop**. The saved `mask.tif` is not loaded, as mask layers are present (use **Load mask** to load it). |
| Masks created after **Crop with main**, then **Reset crop** | The masks are extended to the full image. Regions outside the main ROI are not masked, unless masks were created there before. New masks are created with the full image size and can be combined with the existing ones. |
| Masks created on the full image, used for several main ROIs | Use **shared** masks: changing *Selected main ROI* crops the mask with each main ROI, and saving writes the mask of all main ROIs. With **separate** masks, the masks are saved for the selected main ROI and removed when changing *Selected main ROI*. |
| Main ROI modified after saving the mask | `mask.tif` still corresponds to the old main ROI and is placed at the position of the new one. Masks of a different size are cropped/padded with a warning, but a moved main ROI of the same size gives a shifted mask without warning. Recreate and save the mask after changing a main ROI. |
| Main ROI modified after creating masks, before saving | No problem: the mask is saved for the main ROI as defined at the time of saving. |
| *Selected main ROI* changed after creating masks | Separate masks: the masks are saved for the previously selected main ROI, and the image is cropped with the newly selected one, whose saved mask is loaded. Shared masks: the masks are cropped with the newly selected main ROI. See [Masks of multiple main ROIs](multiple-rois). |
| **Correct and save data** after defining ROIs and masks | The image is reopened from `corrected.zarr`: all layers, including ROIs and masks, are removed and a main ROI covering the full image is added. Define ROIs and masks after the correction. |
| ML mask | The pixel classifier works on `imcube`, i.e. on the current crop, and `ml-mask` has the size of the crop like the other masks. When the crop changes, the annotations memorized for training are discarded, as they refer to the previous crop, but the trained classifier is kept: it can directly be used to segment the new crop or be trained further with new annotations. |

## Use in the other widgets

All other widgets read the project from the project folder: the image from `corrected.zarr`, the ROIs from `Parameters.yml` and the masks from `roi_<n>/mask.tif`. Contrary to the Sediment widget, layers are **not** translated: the viewer coordinates are relative to the top-left corner of the region that is loaded.

```{note}
When several widgets are open in the same viewer, only the widget that last loaded data (opened an image, imported a project, loaded a series) is active. Loading data removes all layers of the viewer, and the other widgets are disabled (except their first tab used to load data) and ignore mouse interactions in the viewer until they load data again. This avoids that layers or settings of one widget interfere with another one, e.g. that masks are shifted or ROIs are added by the wrong widget.
```

### HyperAnalysis

For the main ROI selected in *Selected ROI*, the widget loads the region of the **first sub-ROI** of that main ROI (dimensionality reduction is computationally expensive, so a narrow region is typically used). The mask is cropped from the main ROI mask to the sub-ROI. All outputs (MNF, denoised data, end-members, etc.) correspond to the sub-ROI and are saved in `roi_<n>`.

### SpectralIndices

The widget loads the **main ROI** region and its mask. Index maps are computed over the whole main ROI and are saved without masking. The mask is applied when computing index projections (masked pixels are ignored) and in plots (masked pixels are shown in black in the RGB image and left blank in index images). Index projections are computed by averaging the index map over the columns of the **measurement ROI** (layer `rois`). The measurement ROI is relative to the main ROI. By default, it corresponds to the first sub-ROI. It can be changed by double-clicking in the viewer and is saved in `Parameters.yml`. Index maps (`roi_<n>/index_maps`), projections and plots (`roi_<n>/index_plots`) are saved per main ROI.

### Batch plotting and Concatenation

Batch plotting (SpectralIndices **Batch** tab) uses, for each project and main ROI, the main ROI region of the RGB image, the saved mask and the measurement ROI, as in the SpectralIndices widget. The Concatenation widget assembles the main ROI regions (RGB image and index maps of `roi_0`) of a series of projects.
