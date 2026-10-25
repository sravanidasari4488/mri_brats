# Atlas files for tract meshes

`warp_atlas_tracts` does not download anything. If a file below is missing it raises `AtlasNotFoundError` and lists the path. Copy the files from a local FSL install.

## Layout

```
atlases/MNI152_T1_1mm.nii.gz
atlases/jhu/JHU-ICBM-tracts-prob-2mm.nii.gz
```

Optic radiation is optional. If `atlases/jhu/optic_radiation_L.nii.gz` or `optic_radiation_R.nii.gz` is missing, that tract is skipped and a warning is logged.

The template is FSL's 1 mm MNI152 T1:

```
$FSLDIR/data/standard/MNI152_T1_1mm.nii.gz
```

The 4D file is the JHU white-matter tractography atlas (20 probability volumes, values 0–100):

```
$FSLDIR/data/atlases/JHU/JHU-ICBM-tracts-prob-2mm.nii.gz
```

Volumes are 0-based. Left is the even index of each pair. The code takes the maximum when a tract uses two volumes.

| Tract | Volumes |
| --- | --- |
| Corticospinal tract L / R | 2 / 3 |
| Inferior fronto-occipital fasciculus L / R | 10 / 11 |
| Superior longitudinal fasciculus L / R | 14 / 15 |
| Superior longitudinal fasciculus, temporal part (arcuate) L / R | 18 / 19 |
| Uncinate fasciculus L / R | 16 / 17 |

`arcuate_slf_L` and `arcuate_slf_R` are the maximum of the SLF volume and the temporal SLF volume.

## Optic radiation

`JHU-ICBM-tracts-prob-2mm` has no optic-radiation volume. The ICBM-DTI-81 label atlas contains “posterior thalamic radiation (include optic radiation)”, and that file is a binary parcellation, not a probability map. Do not put it in place of the files below.

Supply a probability map in MNI space for each hemisphere, for example the optic radiation (`or`) written by FSL XTRACT:

```
atlases/jhu/optic_radiation_L.nii.gz
atlases/jhu/optic_radiation_R.nii.gz
```

Either file may be omitted. The missing hemisphere is skipped with a warning, and the tracts that come from the 4D atlas are still warped.

## Individual volumes instead of the 4D file

A 3D file overrides that tract's 4D volumes. Names:

```
atlases/jhu/corticospinal_tract_L.nii.gz
atlases/jhu/corticospinal_tract_R.nii.gz
atlases/jhu/arcuate_slf_L.nii.gz
atlases/jhu/arcuate_slf_R.nii.gz
atlases/jhu/ifof_L.nii.gz
atlases/jhu/ifof_R.nii.gz
atlases/jhu/uncinate_L.nii.gz
atlases/jhu/uncinate_R.nii.gz
atlases/jhu/optic_radiation_L.nii.gz
atlases/jhu/optic_radiation_R.nii.gz
```

To build the arcuate/SLF file yourself (left; volume numbers here are 0-based, so `fslroi` uses the same index):

```
fslroi JHU-ICBM-tracts-prob-2mm.nii.gz slf_L.nii.gz 14 1
fslroi JHU-ICBM-tracts-prob-2mm.nii.gz slf_temp_L.nii.gz 18 1
fslmaths slf_L.nii.gz -max slf_temp_L.nii.gz arcuate_slf_L.nii.gz
```

## What the warp does

SyN registers the MNI152 1 mm T1 to the patient T1. The cost-function mask is the patient brain with the whole tumor removed after a 5-voxel dilation, so the tumor does not drive the warp. Tract maps are warped with linear interpolation.

Maps whose peak is above 1.5 are treated as percent (0–100) and divided by 100. Voxels at or above the probability threshold (default 0.25, i.e. 25%) are meshed. Each tract object has `source="atlas-estimated"`.

Transforms are cached in `caches/tract_transforms/<case_id>/`. A later call for the same case reuses them until the template path or the tumor label changes.

## MNI structural atlas (lobes)

DIPY's PyTorch SynthSeg port does not emit Desikan-Killiany cortex. `parcellate_case` keeps the SynthSeg subcortical labels and warps the FSL MNI structural atlas for the five lobes. It reuses the transforms above. This project does not download the atlas.

```
atlases/MNI152_T1_1mm.nii.gz
atlases/MNI/MNI-maxprob-thr25-1mm.nii.gz
```

The summary image is FSL's 1 mm max-probability MNI structural atlas (threshold 25):

```
$FSLDIR/data/atlases/MNI/MNI-maxprob-thr25-1mm.nii.gz
```

Label values in that file are the XML index plus one: 3 frontal, 4 insula, 5 occipital, 6 parietal, 8 temporal. Those lobes replace SynthSeg cortex (labels 3 and 42). Subcortical SynthSeg labels are left as they are. The warp uses nearest-neighbour interpolation.
