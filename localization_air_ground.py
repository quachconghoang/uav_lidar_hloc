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

images = Path("Data//images_ground/")

outputs = Path("outputs/sfm/")
sfm_pairs = outputs / "pairs-eigenplaces.txt"
sfm_dir = outputs / "sfm_superpoint+superglue"

# retrieval_conf = extract_features.confs["megaloc"]
retrieval_conf = extract_features.confs["eigenplaces"]
feature_conf = extract_features.confs["superpoint_aachen"]
matcher_conf = match_features.confs["superglue"]

retrieval_path = extract_features.main(retrieval_conf, images, outputs)
pairs_from_retrieval.main(retrieval_path, sfm_pairs, num_matched=5)

feature_path = extract_features.main(feature_conf, images, outputs)
match_path = match_features.main(
    matcher_conf, sfm_pairs, feature_conf["output"], outputs
)

model = reconstruction.main(sfm_dir, images, sfm_pairs, feature_path, match_path)

fig = viz_3d.init_figure()
viz_3d.plot_reconstruction(fig, model, color='rgba(255,0,0,0.5)', name="mapping", points_rgb=True)
fig.show()

# For geo-registration, you can use the following command:
# https://colmap.github.io/faq.html#geo-registration