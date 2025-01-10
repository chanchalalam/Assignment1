import cv2

def extract_subobjects(boxes, names, frame):
    objects = []
    sub_objects = []

    for idx, bbox in enumerate(boxes):
        x1, y1, x2, y2 = map(int, bbox)
        obj_name = names[idx]

        if obj_name in ["person"]:  # Main object
            objects.append({
                "object": obj_name,
                "id": idx + 1,
                "bbox": [x1, y1, x2, y2]
            })
        elif obj_name in ["helmet"]:  # Example sub-object
            sub_objects.append({
                "object": obj_name,
                "id": idx + 1,
                "bbox": [x1, y1, x2, y2]
            })

    return objects, sub_objects


def format_json(objects, sub_objects):
    detections = []
    for obj in objects:
        detection = {
            "object": obj["object"],
            "id": obj["id"],
            "bbox": obj["bbox"],
            "subobject": []
        }

        for sub_obj in sub_objects:
            if (
                sub_obj["bbox"][0] >= obj["bbox"][0]
                and sub_obj["bbox"][1] >= obj["bbox"][1]
                and sub_obj["bbox"][2] <= obj["bbox"][2]
                and sub_obj["bbox"][3] <= obj["bbox"][3]
            ):
                detection["subobject"].append({
                    "object": sub_obj["object"],
                    "id": sub_obj["id"],
                    "bbox": sub_obj["bbox"]
                })

        detections.append(detection)

    return detections


def save_subobject_images(objects, sub_objects, frame, output_dir, frame_number):
    for obj in objects:
        x1, y1, x2, y2 = map(int, obj['bbox'])
        cropped_image = frame[y1:y2, x1:x2]
        filename = f"{output_dir}/frame_{frame_number}_object_{obj['id']}.jpg"
        cv2.imwrite(filename, cropped_image)
        print(f"Saved object image: {filename}")

    for sub_obj in sub_objects:
        x1, y1, x2, y2 = map(int, sub_obj['bbox'])
        cropped_image = frame[y1:y2, x1:x2]
        filename = f"{output_dir}/frame_{frame_number}_subobject_{sub_obj['id']}.jpg"
        cv2.imwrite(filename, cropped_image)
        print(f"Saved sub-object image: {filename}")
