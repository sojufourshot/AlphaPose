from inference import inference


def CCW(a, b):
    """CCW algorithm

    Args:
        a (vector2D): vector
        b (vector2D): vector

    Returns:
        number: CCW scalar value
    """
    x1, y1 = a
    x2, y2 = b
    return x1*y2-x2*y1


def normalization(joints, bbox):
    """Normalize joints

    Args:
        joints (List[vector2D]): some of joint vectors
        bbox (List[number]): bounding box

    Returns:
        List[vector2D]: normalized joint vectors
    """
    left, bottom, width, height = bbox
    new_joints = []
    for x, y in joints:
        x -= left
        y -= bottom
        new_joints.append(x/width, y/height)
    return new_joints


def get_feature(img):
    """Run AI model

    Args:
        img (string): image path

    Returns:
        object: name, keypoints, bboxes
    """
    result = inference(img)[0]
    # assert 0
    name = result["imgname"]
    result = result["result"]
    keypoints = []
    bboxes = []
    proposal_scores = []
    for item in result:
        keypoint = item["keypoints"]
        bbox = item["bbox"]  # [left, bottom, width, height]
        proposal_score = item["proposal_score"]
        keypoints.append(keypoint.tolist())
        bboxes.append(bbox)
        proposal_scores.append(proposal_score)
    ix = 0
    for i in range(len(proposal_score)):
        if proposal_score[i] > proposal_score[ix]:
            ix = i
    return name, keypoints[ix], bboxes[ix]


def get_score(img1, img2):
    pass


if __name__ == "__main__":
    print(get_feature("examples/demo/1.jpg"))
