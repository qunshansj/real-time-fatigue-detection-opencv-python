
def eye_aspect_ratio(eye):
    # 计算距离，竖直的
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    # 计算距离，水平的
    C = dist.euclidean(eye[0], eye[3])
    # ear值
    ear = (A + B) / (2.0 * C)
    return ear

