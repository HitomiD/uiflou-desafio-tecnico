import json

class FrameDataExporter:
    """
    Handles exporting object detection and pose data to a JSONL file.
    Each call to export_frame() appends one JSON object per frame.
    """
    def __init__(self, output_path, keypoint_names=None):
        self.output_path = output_path
        self.keypoint_names = keypoint_names or [
            "nose", "left_eye", "right_eye", "left_ear", "right_ear",
            "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
            "left_wrist", "right_wrist", "left_hip", "right_hip",
            "left_knee", "right_knee", "left_ankle", "right_ankle"
        ]

    def export_frame(self, frame_number, results, model_names):
        frame_data = {
            "frame": frame_number,
            "objects": [],
            "keypoints": []
        }

        # Extract bounding boxes
        for idx, box in enumerate(results[0].boxes):
            bbox = box.xyxy.tolist()
            confidence = float(box.conf)
            class_id = int(box.cls)
            class_name = model_names[class_id]

            frame_data["objects"].append({
                "id": idx,
                "class": class_name,
                "bbox": bbox,
                "confidence": confidence
            })

        # Extract keypoints (pose)
        if hasattr(results[0], "keypoints") and results[0].keypoints is not None:
            # FIX APPLIED HERE: Access the raw tensor data using .data
            keypoints_data_tensor = results[0].keypoints.data 
            
            # keypoints_data_tensor shape is (Num_People, Num_Keypoints, 3)
            for person_id, kp_data in enumerate(keypoints_data_tensor):
                # kp_data is now a tensor of shape (Num_Keypoints, 3)
                
                # Convert tensor to 2D numpy array of shape (num_keypoints, 3)
                try:
                    # These methods are valid on a torch.Tensor
                    kp_array = kp_data.cpu().numpy().reshape(-1, 3)
                except AttributeError:
                    # Fallback
                    kp_array = kp_data.reshape(-1, 3)
                    
                keypoints_list = []
                for kp_idx, (x, y, c) in enumerate(kp_array):
                    keypoints_list.append({
                        "name": self.keypoint_names[kp_idx],
                        "x": float(x),
                        "y": float(y),
                        "confidence": float(c)
                    })
                frame_data["keypoints"].append({
                    "person_id": person_id,
                    "keypoints": keypoints_list
                })

        # Append to JSONL file
        with open(self.output_path, "a") as f:
            f.write(json.dumps(frame_data) + "\n")