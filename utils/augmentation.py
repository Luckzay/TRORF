import cv2
import numpy as np

class DataAugmentation:
    def __init__(self):
        pass

    @staticmethod
    def _blend(fg, bg, alpha):               # 辅助：两图混合
        return (alpha * fg + (1 - alpha) * bg).astype(np.uint8)

    @staticmethod
    def brightness_shift(img, limit=0.08):   # 1. 亮度漂移
        """limit: 0.08 ≈ ±20/255"""
        beta = np.random.uniform(-limit, limit) * 255
        return cv2.convertScaleAbs(img, beta=beta)

    @staticmethod
    def gamma_adjust(img, gamma_range=(0.8, 1.2)):  # 2. 伽马校正
        gamma = np.random.uniform(*gamma_range)
        inv = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv) * 255 for i in range(256)])
        return cv2.LUT(img, table.astype(np.uint8))

    @staticmethod
    def channel_swap(img, p=0.5):            # 3. 随机交换 R/G/B 两通道
        if np.random.rand() > p:
            return img
        rg = img[:,:,[1,0,2]]
        rb = img[:,:,[2,1,0]]
        gb = img[:,:,[0,2,1]]
        choice = np.random.choice(['rg', 'rb', 'gb'])
        if choice == 'rg':
            return rg
        elif choice == 'rb':
            return rb
        else:
            return gb

    @staticmethod
    def color_jitter(img, hue_delta=3, sat_scale=(0.85, 1.15)):
        # 4. 仅 HSV 空间微小色相/饱和度抖动
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
        # hue
        hsv[:,:,0] += np.random.randint(-hue_delta, hue_delta+1)
        hsv[:,:,0] = np.clip(hsv[:,:,0], 0, 179)
        # saturation
        sat_factor = np.random.uniform(*sat_scale)
        hsv[:,:,1] *= sat_factor
        hsv[:,:,1] = np.clip(hsv[:,:,1], 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

    @staticmethod
    def add_shade(img, max_alpha=0.25):      # 5. 模拟杯壁阴影/反光
        h, w = img.shape[:2]
        shade = np.random.uniform(0.75, 1.0, (h, w, 1))
        shade = np.tile(shade, [1,1,3])
        return DataAugmentation._blend(img, (shade*255).astype(np.uint8), max_alpha)

    @staticmethod
    def random_flip_rotate(img, max_angle=5): # 6. 轻旋转 + 随机水平翻转
        angle = np.random.uniform(-max_angle, max_angle)
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
        rotated = cv2.warpAffine(img, M, (w, h),
                                 borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=(0,0,0))
        if np.random.rand() < 0.5:
            rotated = cv2.flip(rotated, 1)
        return rotated

    @staticmethod
    def gaussian_noise(img, mean=0, std=0.1):
        """7. 添加高斯噪声"""
        noise = np.random.normal(mean, std, img.shape) * 255
        noisy_img = img + noise
        return np.clip(noisy_img, 0, 255).astype(np.uint8)

    @staticmethod
    def motion_blur(img, degree=5, angle=0):
        """8. 运动模糊"""
        angle = np.random.uniform(-360, 360) if angle == 0 else angle
        M = cv2.getRotationMatrix2D((degree/2, degree/2), angle, 1)
        motion_blur_kernel = np.diag(np.ones(degree))
        motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, M, (degree, degree))

        motion_blur_kernel = motion_blur_kernel / degree
        blurred = cv2.filter2D(img, -1, motion_blur_kernel)
        return blurred

    @staticmethod
    def contrast_adjust(img, alpha_range=(0.8, 1.2)):
        """9. 对比度调整"""
        alpha = np.random.uniform(*alpha_range)
        return cv2.convertScaleAbs(img, alpha=alpha, beta=0)

    # 新增高级数据增强方法
    @staticmethod
    def cutout(img, mask_size=32, p=0.5):
        """10. Cutout 数据增强 - 随机遮挡图像的一部分"""
        if np.random.rand() > p:
            return img

        h, w = img.shape[:2]
        cx = np.random.randint(0, w)
        cy = np.random.randint(0, h)

        x1 = np.clip(cx - mask_size // 2, 0, w)
        y1 = np.clip(cy - mask_size // 2, 0, h)
        x2 = np.clip(cx + mask_size // 2, 0, w)
        y2 = np.clip(cy + mask_size // 2, 0, h)

        img[y1:y2, x1:x2] = 0
        return img

    @staticmethod
    def random_erase(img, probability=0.5, sl=0.02, sh=0.4, r1=0.3, r2=3.3, pixel_level=True):
        """11. Random Erasing - 随机擦除图像区域"""
        if np.random.rand() > probability:
            return img

        h, w = img.shape[:2]
        area = h * w

        for attempt in range(100):
            target_area = np.random.uniform(sl, sh) * area
            aspect_ratio = np.random.uniform(r1, r2)

            h_obstruct = int(round(np.sqrt(target_area * aspect_ratio)))
            w_obstruct = int(round(np.sqrt(target_area / aspect_ratio)))

            if w_obstruct < w and h_obstruct < h:
                x1 = np.random.randint(0, w - w_obstruct)
                y1 = np.random.randint(0, h - h_obstruct)

                if pixel_level:
                    if len(img.shape) == 3:
                        for i in range(img.shape[2]):
                            img[y1:y1+h_obstruct, x1:x1+w_obstruct, i] = np.random.randint(0, 255)
                    else:
                        img[y1:y1+h_obstruct, x1:x1+w_obstruct] = np.random.randint(0, 255)
                else:
                    img[y1:y1+h_obstruct, x1:x1+w_obstruct] = np.random.randint(0, 255)
                return img

        return img

    @staticmethod
    def mixup(img1, img2, alpha=0.2):
        """12. Mixup - 将两张图像按一定比例混合"""
        lam = np.random.beta(alpha, alpha)
        mixed_img = lam * img1 + (1 - lam) * img2
        return mixed_img.astype(np.uint8)

    @staticmethod
    def gridmask(img, d_ratio=0.6, ratio=0.5, rotate=1, mode=0, p=0.5):
        """13. GridMask - 在图像上应用网格掩码"""
        if np.random.rand() > p:
            return img

        h, w = img.shape[:2]

        hh = int(np.ceil(np.sqrt(h*h+w*w)))
        ww = hh

        # 创建网格掩码
        d = np.random.randint(int(d_ratio * min(h, w)), min(h, w))
        l = int(d * ratio)

        mask = np.ones((hh, ww)).astype(np.float32)

        # 创建网格
        for i in range(hh // d):
            s = d*i
            t = min(s + l, hh)
            mask[s:t, :] = 0

        # 旋转
        if rotate > 0:
            angle = np.random.randint(-rotate, rotate)
            center = (hh // 2, ww // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1)
            mask = cv2.warpAffine(mask, M, (hh, ww), flags=cv2.INTER_NEAREST)

        # 调整大小到原图尺寸
        mask = mask[int((hh-h)/2):int((hh-h)/2)+h, int((ww-w)/2):int((ww-w)/2)+w]
        mask = np.expand_dims(mask, axis=-1) if len(img.shape) == 3 else mask

        return (img * mask).astype(np.uint8)

    @staticmethod
    def elastic_transform(img, alpha=2000, sigma=40, p=0.5):
        """14. Elastic Transform - 弹性变形"""
        if np.random.rand() > p:
            return img

        h, w = img.shape[:2]

        dx = cv2.GaussianBlur((np.random.rand(h, w) * 2 - 1), (0, 0), sigma) * alpha
        dy = cv2.GaussianBlur((np.random.rand(h, w) * 2 - 1), (0, 0), sigma) * alpha

        x, y = np.meshgrid(np.arange(w), np.arange(h))
        mapx = np.float32(x + dx)
        mapy = np.float32(y + dy)

        return cv2.remap(img, mapx, mapy, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    @staticmethod
    def resize_and_pad(img, target_size=(224, 224)):
        """15. Resize and Pad - 调整大小并填充"""
        h, w = img.shape[:2]
        target_h, target_w = target_size

        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)

        resized_img = cv2.resize(img, (new_w, new_h))

        padded_img = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        pad_x = (target_w - new_w) // 2
        pad_y = (target_h - new_h) // 2

        padded_img[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = resized_img
        return padded_img

    @staticmethod
    def cutmix(img1, bbox1, img2, bbox2, p=0.5):
        """16. CutMix - 结合两张图像和标签"""
        if np.random.rand() > p:
            return img1, bbox1

        h, w = img1.shape[:2]

        # 随机确定裁剪区域
        bbx1 = np.random.randint(0, w)
        bby1 = np.random.randint(0, h)
        bbw = np.random.randint(0, w - bbx1)
        bbh = np.random.randint(0, h - bby1)

        # 将img2的部分区域粘贴到img1上
        new_img = img1.copy()
        new_img[bby1:bby1+bbh, bbx1:bbx1+bbw] = img2[bby1:bby1+bbh, bbx1:bbx1+bbw]

        # 更新边界框
        new_bbox = np.concatenate([bbox1, bbox2])
        return new_img, new_bbox

    @staticmethod
    def autoaugment_policy(img):
        """17. AutoAugment Policy - 自动增强策略示例"""
        policies = [
            lambda x: DataAugmentation.brightness_shift(x, limit=0.1),
            lambda x: DataAugmentation.contrast_adjust(x, alpha_range=(0.7, 1.3)),
            lambda x: DataAugmentation.color_jitter(x, hue_delta=5, sat_scale=(0.8, 1.2)),
            lambda x: DataAugmentation.motion_blur(x, degree=3),
            lambda x: DataAugmentation.gaussian_noise(x, std=0.05),
            lambda x: DataAugmentation.random_flip_rotate(x, max_angle=10)
        ]

        policy = np.random.choice(policies)
        return policy(img)

    @staticmethod
    def rand_augment(img, num_ops=2, magnitude=9):
        """18. RandAugment - 随机增强"""
        ops = [
            DataAugmentation.brightness_shift,
            DataAugmentation.contrast_adjust,
            DataAugmentation.gamma_adjust,
            DataAugmentation.color_jitter,
            DataAugmentation.motion_blur,
            DataAugmentation.gaussian_noise,
            DataAugmentation.random_flip_rotate
        ]

        augmented_img = img.copy()
        for _ in range(num_ops):
            op = np.random.choice(ops)
            augmented_img = op(augmented_img)

        return augmented_img

    @staticmethod
    def apply_augmentation(img, augmentation_types=None):
        """应用指定的数据增强组合"""
        if augmentation_types is None:
            # 默认使用几种增强方式的组合
            augmentation_types = ['brightness_shift', 'gamma_adjust', 'random_flip_rotate']

        augmented_img = img.copy()

        for aug_type in augmentation_types:
            if hasattr(DataAugmentation, aug_type):
                aug_method = getattr(DataAugmentation, aug_type)
                augmented_img = aug_method(augmented_img)

        return augmented_img

    @staticmethod
    def get_default_pipeline():
        """获取默认的增强流水线"""
        return ['brightness_shift', 'gamma_adjust', 'color_jitter', 'random_flip_rotate']

    @staticmethod
    def apply_random_augmentation(img, num_augs=None, exclude=None):
        """随机应用指定数量的数据增强"""
        if exclude is None:
            exclude = []

        # 所有可能的增强类型
        all_augs = [
            'brightness_shift', 'gamma_adjust', 'channel_swap',
            'color_jitter', 'add_shade', 'random_flip_rotate',
            'gaussian_noise', 'motion_blur', 'contrast_adjust',
            'cutout', 'random_erase', 'gridmask', 'elastic_transform',
            'autoaugment_policy', 'rand_augment'
        ]

        # 排除不需要的增强类型
        available_augs = [aug for aug in all_augs if aug not in exclude]

        # 随机选择增强数量
        if num_augs is None:
            num_augs = np.random.randint(1, min(4, len(available_augs)+1))

        # 随机选择增强类型并应用
        selected_augs = np.random.choice(available_augs, size=num_augs, replace=False)

        augmented_img = img.copy()
        for aug_type in selected_augs:
            aug_method = getattr(DataAugmentation, aug_type)
            augmented_img = aug_method(augmented_img)

        return augmented_img