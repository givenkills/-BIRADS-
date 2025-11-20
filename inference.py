from ultralytics import YOLO
import cv2
import os
import numpy as np


def show_batch_results(images, titles, num_cols=3):
    """按网格显示多张图片"""
    num_images = len(images)
    num_rows = (num_images + num_cols - 1) // num_cols

    figure_size = 1500
    canvas = np.zeros((figure_size * num_rows, figure_size * num_cols, 3), dtype=np.uint8)

    for idx, (img, title) in enumerate(zip(images, titles)):
        row = idx // num_cols
        col = idx % num_cols

        h, w = img.shape[:2]
        scale = min(figure_size / w, figure_size / h)
        new_w, new_h = int(w * scale), int(h * scale)
        # 使用更好的插值方法
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        x_offset = col * figure_size + (figure_size - new_w) // 2
        y_offset = row * figure_size + (figure_size - new_h) // 2

        canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

        # 优化标题显示
        title = os.path.splitext(title)[0]
        cv2.putText(canvas, title, (x_offset, y_offset - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return canvas


def draw_label(img, text, pos):
    """优化的标签绘制函数"""
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 1
    padding = 5

    # 获取文本大小
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)

    x, y = pos
    img_h, img_w = img.shape[:2]

    # 确保标签在图像范围内
    if x + text_w + 2 * padding > img_w:
        x = img_w - text_w - 2 * padding
    if y - text_h - 2 * padding < 0:
        y = text_h + 2 * padding

    # 绘制白色背景
    cv2.rectangle(img,
                  (int(x), int(y - text_h - 2 * padding)),
                  (int(x + text_w + 2 * padding), int(y)),
                  (255, 255, 255), -1)

    # 绘制文本
    cv2.putText(img, text,
                (int(x + padding), int(y - padding)),
                font, font_scale, (0, 0, 0), thickness,
                cv2.LINE_AA)  # 使用 LINE_AA 获得更平滑的文本


def main():
    # 加载模型
    model_path =  r"D:\DIGIX-main\DIGIX-main\ultralytics-main\runs\classify\train10\weights\best.pt" # 替换为你的模型路径
    model = YOLO(model_path)

    # 设置数据集路径
    dataset_path = r"C:\乳腺训练数据集\乳腺训练数据集\test_A"  # 替换为你的数据集路径

    image_files = [f for f in os.listdir(dataset_path)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    batch_size = 6

    for i in range(0, len(image_files), batch_size):
        batch_images = image_files[i:i + batch_size]
        processed_images = []
        titles = []

        for img_name in batch_images:
            img_path = os.path.join(dataset_path, img_name)
            # 读取高质量图像
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

            results = model.predict(img, show_conf=True)[0]

            annotated_img = img.copy()
            if results.boxes is not None:
                boxes = results.boxes.cpu().numpy()
                for box in boxes:
                    # 获取边界框坐标
                    x1, y1, x2, y2 = box.xyxy[0]
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    class_name = results.names[cls]

                    # 绘制边界框
                    cv2.rectangle(annotated_img,
                                  (int(x1), int(y1)),
                                  (int(x2), int(y2)),
                                  (0, 255, 0), 2)

                    # 显示类别名称和置信度
                    label = f"{class_name} {conf:.2f}"
                    draw_label(annotated_img, label, (int(x1), int(y1)))

            processed_images.append(annotated_img)
            titles.append(img_name)

        canvas = show_batch_results(processed_images, titles)

        window_name = 'Batch Results'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 900, 600)
        cv2.imshow(window_name, canvas)

        # 添加键盘控制
        key = cv2.waitKey(0)
        if key == ord('q'):  # 按'q'退出
            break
        elif key == ord('s'):  # 按's'保存结果
            save_path = f'batch_result_{i // batch_size}.jpg'
            # 保存高质量图像
            cv2.imwrite(save_path, canvas, [cv2.IMWRITE_JPEG_QUALITY, 100])
            print(f"Saved result to {save_path}")
        elif key == ord('n'):  # 按'n'显示下一批
            continue

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()