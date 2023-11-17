
# 遍历每一帧
while True:
    # 预处理
    frame = vs.read()[1]
    if frame is None:
        break

    (h, w) = frame.shape[:2]
    width=1200
    r = width / float(w)
    dim = (width, int(h * r))
    frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 检测人脸
    rects = detector(gray, 0)

    # 遍历每一个检测到的人脸
    for rect in rects:
        # 获取坐标
        shape = predictor(gray, rect)
        shape = shape_to_np(shape)

        # 分别计算ear值
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)

        # 算一个平均的
        ear = (leftEAR + rightEAR) / 2.0

        # 绘制眼睛区域
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        # 检查是否满足阈值
        if ear < EYE_AR_THRESH:
            COUNTER += 1

        else:
            # 如果连续几帧都是闭眼的，总数算一次
            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                TOTAL += 1

            # 重置
            COUNTER = 0

        # 显示
        cv2.putText(frame, "Blinks: {}".format(TOTAL), (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(10) & 0xFF

    if key == 27:
        break

vs.release()
cv2.destroyAllWindows()
