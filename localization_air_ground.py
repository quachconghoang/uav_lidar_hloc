#%%
from pathlib import Path

from hloc import (
    extract_features,
    match_features,
    reconstruction,
    visualization,
    pairs_from_retrieval,
)

from hloc.utils import viz_3d
import pycolmap

images = Path("Data/images_ground/")

outputs = Path("outputs/sfm/")
sfm_pairs = outputs / "pairs-eigenplaces.txt"
sfm_dir = outputs / "sfm_superpoint+superglue"

retrieval_conf = extract_features.confs["megaloc"]
# retrieval_conf = extract_features.confs["eigenplaces"]
feature_conf = extract_features.confs["superpoint_aachen"]
matcher_conf = match_features.confs["superglue"]

# %% finding pairs from retrieval

retrieval_path = extract_features.main(retrieval_conf, images, outputs)
pairs_from_retrieval.main(retrieval_path, sfm_pairs, num_matched=5)

feature_path = extract_features.main(feature_conf, images, outputs)
match_path = match_features.main(
    matcher_conf, sfm_pairs, feature_conf["output"], outputs
)

# %% running reconstruction
model = reconstruction.main(sfm_dir, images, sfm_pairs, feature_path, match_path)

# %% reloading reconstruction if sfm_dir is not empty
# model = pycolmap.Reconstruction(sfm_dir)

'''
colmap model_aligner \
    --input_path ./sfm_superpoint+superglue \
    --output_path ./sfm_scaled \
    --database_path ./sfm_superpoint+superglue/database.db \ (--ref_images_path /path/to/text-file)
    --ref_is_gps 1 \
    --alignment_type enu \
    --alignment_max_error 3.0
'''

model = pycolmap.Reconstruction(outputs/'sfm_scaled')

# %% visualization
import plotly.graph_objects as go

fig = viz_3d.init_figure()
fig.update_layout(
    template="plotly_white"
)

# draw Ox Oy Oz axes
fig.add_trace(go.Scatter3d(x=[0, 10], y=[0, 0], z=[0, 0], mode='lines',
                           line=dict(color='red', width=4), name='X-axis'    ),)
fig.add_trace(go.Scatter3d(x=[0, 0], y=[0, 10], z=[0, 0], mode='lines',
                           line=dict(color='green', width=4), name='Y-axis'    ),)
fig.add_trace(go.Scatter3d(x=[0, 0], y=[0, 0], z=[0, 10], mode='lines',
                           line=dict(color='blue', width=4), name='Z-axis'   ),)


#
viz_3d.plot_reconstruction(fig, model, color='rgba(255,0,0,0.5)', name="mapping", points_rgb=True)

fig.show()

# For geo-registration, you can use the following command:
# https://colmap.github.io/faq.html#geo-registration

# manhattan world alighnment

#%% rename images in the Data/images_gnd2/
images_dir = Path("Data/images_ground/")
img_list = list(images_dir.glob("*.JPG"))
img_list.sort()

for i, img in enumerate(img_list):
    new_name = f"{i:04d}.JPG"
    img.rename(images_dir / new_name)
    print(f"Renamed {img.name} to {new_name}")