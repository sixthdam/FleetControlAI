import os

import fiftyone as fo
import fiftyone.zoo as foz


MANHATTAN_BOUNDARY = [
    [
        [-73.949701, 40.834487],
        [-73.896611, 40.815076],
        [-73.998083, 40.696534],
        [-74.031751, 40.715273],
        [-73.949701, 40.834487],
    ]
]


def load_fleet_data():
    fo.config.database_uri = os.getenv("FIFTYONE_DATABASE_URI")
    dataset = foz.load_zoo_dataset("quickstart-geo")
    dataset.persistent = True

    try:
        dataset.create_index([("location.point", "2dsphere")])
    except Exception:
        pass

    if "num_objects" not in dataset.get_field_schema():
        dataset.add_sample_field("num_objects", fo.IntField)
        for sample in dataset.iter_samples(autosave=True, progress=True):
            sample["num_objects"] = (
                len(sample.ground_truth.detections)
                if sample.ground_truth
                else 0
            )

    if "detection_classes" not in dataset.get_field_schema():
        dataset.add_sample_field("detection_classes", fo.ListField)
        for sample in dataset.iter_samples(autosave=True, progress=True):
            if sample.ground_truth:
                sample["detection_classes"] = list(
                    set(d.label for d in sample.ground_truth.detections)
                )

    return dataset


def fleet_stats(dataset):
    total_objects = sum(
        s.ground_truth and len(s.ground_truth.detections) or 0
        for s in dataset
    )
    return {
        "samples": len(dataset),
        "total_detections": total_objects,
        "avg_objects_per_sample": round(total_objects / len(dataset), 1),
    }


def geo_fence_analysis(dataset, boundary=MANHATTAN_BOUNDARY):
    inside = dataset.geo_within(boundary, location_field="location")
    outside = dataset.exclude(inside)
    return {"inside": inside, "outside": outside, "num_inside": len(inside), "num_outside": len(outside)}


def near_point_analysis(dataset, point, meters=5000):
    near = dataset.geo_near(point, max_distance=meters, location_field="location")
    return near


def object_class_summary(dataset):
    from collections import Counter

    class_counts = Counter()
    for sample in dataset:
        if sample.ground_truth:
            for d in sample.ground_truth.detections:
                class_counts[d.label] += 1
    return class_counts
