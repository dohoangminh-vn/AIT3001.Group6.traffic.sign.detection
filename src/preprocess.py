"""Online image geometry shared by backends that need explicit transforms."""

def letterbox_geometry(width, height, size=640):
    """Return scale and centered padding for aspect-preserving square resize."""
    if min(width, height, size) <= 0:
        raise ValueError("Image dimensions and target size must be positive")
    scale = min(size / width, size / height)
    resized_width = max(1, min(size, int(round(width * scale))))
    resized_height = max(1, min(size, int(round(height * scale))))
    left, top = (size - resized_width) // 2, (size - resized_height) // 2
    return {"scale_x": resized_width / width, "scale_y": resized_height / height,
            "resized_width": resized_width, "resized_height": resized_height,
            "pad_left": left, "pad_top": top,
            "pad_right": size - resized_width - left,
            "pad_bottom": size - resized_height - top}


def inverse_bbox_xyxy(bbox, geometry, width, height):
    """Map a letterboxed xyxy box back to original-image coordinates."""
    x1, y1, x2, y2 = map(float, bbox)
    x1 = (x1 - geometry["pad_left"]) / geometry["scale_x"]
    x2 = (x2 - geometry["pad_left"]) / geometry["scale_x"]
    y1 = (y1 - geometry["pad_top"]) / geometry["scale_y"]
    y2 = (y2 - geometry["pad_top"]) / geometry["scale_y"]
    return [max(0.0, min(width, x1)), max(0.0, min(height, y1)),
            max(0.0, min(width, x2)), max(0.0, min(height, y2))]


def online_geometry_transforms(size):
    """Return the shared online aspect-preserving resize and centered pad steps."""
    try:
        import albumentations as A
    except ImportError as error:
        raise RuntimeError("Install requirements.txt to use online Albumentations transforms") from error
    return [
        A.LongestMaxSize(max_size=int(size), interpolation=1, p=1.0),
        A.PadIfNeeded(min_height=int(size), min_width=int(size), position="center",
                      border_mode=0, fill=(114, 114, 114), p=1.0),
    ]


def make_online_transform(size, bbox_params=None):
    """Albumentations letterbox transform; gray fill is an implementation detail."""
    import albumentations as A
    transforms = online_geometry_transforms(size)
    return A.Compose(transforms, bbox_params=bbox_params) if bbox_params else A.Compose(transforms)


