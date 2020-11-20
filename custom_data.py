from detectron2.data.datasets import register_coco_instances
from detectron2.data.datasets import register_pascal_voc
from detectron2.data import DatasetCatalog, MetadataCatalog
import os

# CLASS_NAMES1 = ("nohat","helmets","mask","nomask",)
# CLASS_NAMES = ("hat","person",)
CLASS_NAMES = ("helmets","nohat","mask","nomask","vest")

SPLITS = [
        ("custom", "HF", "train"),
    ]

for name, dirname, split in SPLITS:
    year = 2007 if "2007" in name else 2012
    register_pascal_voc(name, os.path.join("./", dirname), split, year, class_names=CLASS_NAMES)
    # MetadataCatalog.get(name).evaluator_type = "pascal_voc"
