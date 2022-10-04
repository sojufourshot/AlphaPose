if __name__ == "__main__":
    from inference import inference
else:
    from .inference import inference


def cosine_similarity(a, b):
    """Get cosine similarity

    Args:
        a (Point): vector1
        b (Point): vector2

    Returns:
        value: cosine value between two vectors
    """
    x1, y1 = a
    x2, y2 = b
    dot_product = x1*x2+y1*y2
    na = (x1**2+y1**2)**0.5
    nb = (x2**2+y2**2)**0.5
    return dot_product / (na*nb)


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
        new_joints.append([x/width, y/height])
    return new_joints


def get_feature(img):
    """Run AI model
       get keypoints, boundingbox

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
        keypoint = keypoint.tolist()

        # 가슴 좌표 구하기 (양 어깨의 중심)
        x1, y1 = keypoint[5]  # left shoulder
        x2, y2 = keypoint[6]  # right shoulder
        keypoint.append([(x1+x2)/2, (y1+y2)/2])

        keypoints.append(keypoint)
        bboxes.append(bbox)
        proposal_scores.append(proposal_score)
    ix = 0
    for i in range(len(proposal_score)):  # best confidence person
        if proposal_score[i] > proposal_score[ix]:
            ix = i
    return name, keypoints[ix], bboxes[ix]


def get_joint_vector(keypoints):
    """관절을 잇는 뼈대 벡터 반환

    Args:
        keypoints (List[Point]): 관절들의 리스트

    Returns:
        Joint vector: 인접한 관절을 이은 뼈대 벡터의 리스트
    """
    # adjacent keypoint
    directions = [[0, 17], [17, 6], [17, 5], [6, 8], [8, 10], [5, 7], [
        7, 9], [17, 12], [12, 14], [14, 16], [17, 11], [11, 13], [13, 15]]
    vectors = []
    for p1, p2 in directions:
        x1, y1 = keypoints[p1]
        x2, y2 = keypoints[p2]
        vectors.append([x2-x1, y2-y1])
    return vectors


def processing(img: str):
    """이미지로부터 뼈대 벡터 구하는 함수

    Args:
        img (str): 이미지 경로

    Returns:
        Joint vector: 뼈대 벡터 리스트
    """
    name, keypoint, bbox = get_feature(img)
    normalized_keypoint = normalization(keypoint, bbox)
    joint = get_joint_vector(normalized_keypoint)
    return joint


def get_score(img1: str, img2: str):
    """자세 점수 구하는 함수

    Args:
        img1 (str): 원본 이미지 경로
        img2 (str): 사용자 이미지 경로

    Returns:
        score: 유사도 점수
    """
    joint1 = processing(img1)
    joint2 = processing(img2)
    length = len(joint1)
    score = 0
    for v1, v2 in zip(joint1, joint2):
        score += (cosine_similarity(v1, v2) + 1)*50  # 0 <= x <= 100
        # print(round((cosine_similarity(v1, v2) + 1)*50, 2))
    score /= length
    return score


if __name__ == "__main__":
    # value = processing("examples/demo/4.jpg")
    score = get_score("examples/demo/12.jpg", "examples/demo/13.jpg")
    print(score)
