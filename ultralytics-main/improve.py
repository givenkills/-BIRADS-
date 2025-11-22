from PIL import Image, ImageEnhance, ImageOps
import os

def data_augmentation(input_folder, output_folder):
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历输入文件夹中的所有图片
    for image_name in os.listdir(input_folder):
        if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(input_folder, image_name)
            image = Image.open(image_path)

            # 旋转
            rotated_image = image.rotate(45, expand=True)
            rotated_image.save(os.path.join(output_folder, f'rotated_{image_name}'))

            # 翻转
            flipped_image = ImageOps.mirror(image)
            flipped_image.save(os.path.join(output_folder, f'flipped_{image_name}'))

            # 调整亮度
            enhancer = ImageEnhance.Brightness(image)
            enhanced_image = enhancer.enhance(1.5)  # 增加50%的亮度
            enhanced_image.save(os.path.join(output_folder, f'brighter_{image_name}'))

            # 调整对比度
            enhancer = ImageEnhance.Contrast(image)
            enhanced_image = enhancer.enhance(1.5)  # 增加50%的对比度
            enhanced_image.save(os.path.join(output_folder, f'higher_contrast_{image_name}'))

if __name__ == "__main__":
    input_folder = r'C:\Users\ZhiYi\OneDrive\Desktop\ultralytics-main\datasets\cancer\train\6'  # 输入文件夹路径
    output_folder = r'C:\Users\ZhiYi\OneDrive\Desktop\photo-improve\train\6'  # 输出文件夹路径
    data_augmentation(input_folder, output_folder)