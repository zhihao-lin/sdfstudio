from pathlib import Path, PurePath
from tqdm import tqdm
import numpy as np
from PIL import Image
def _generate_dataparser_outputs(root_dir):

    all_frame_names = sorted([x.stem for x in (root_dir / "color").iterdir() if x.name.endswith('.jpg')],
                                  key=lambda y: int(y) if y.isnumeric() else y)
    sample_indices = list(range(len(all_frame_names)))


    Path(root_dir / "valid").mkdir(exist_ok=True)
    for sample_index in tqdm(sample_indices):
        mask_path = root_dir / "invalid" / f"{all_frame_names[sample_index]}.jpg"
        valid_mask_path = root_dir / "valid" / f"{all_frame_names[sample_index]}.jpg"
        assert mask_path.exists()
        mask = np.array(Image.open(mask_path), dtype=np.uint8)
        mask = 255 - mask
        mask = Image.fromarray(mask)
        mask.save(valid_mask_path)

_generate_dataparser_outputs(Path("/u/xia1/codes/psdf/data/plift"))