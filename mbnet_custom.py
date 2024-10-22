import cv2
import imutils
import os
import csv
import random
import torch
import torch.nn as nn
import torch.nn.functional as F


# Differential Modality Aware Fusion (DMAF) module
class DMAFModule(nn.Module):
    def __init__(self, in_channels):
        super(DMAFModule, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // 4, in_channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, rgb_feat, thermal_feat):
        diff_feat = rgb_feat - thermal_feat
        global_diff = self.global_avg_pool(diff_feat)
        weights = self.fc(global_diff)
        modulated_rgb = rgb_feat * weights
        modulated_thermal = thermal_feat * (1 - weights)
        dmaf_output = modulated_rgb + modulated_thermal
        dmaf_output = dmaf_output * self.gamma + rgb_feat * (1 - self.gamma)
        return dmaf_output

# Illumination Aware Feature Alignment (IAFA) module
class IAFAAlignmentModule(nn.Module):
    def __init__(self):
        super(IAFAAlignmentModule, self).__init__()
        self.rgb_conv = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.thermal_conv = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.alpha = nn.Parameter(torch.ones(1))

    def forward(self, rgb_img, thermal_img):
        rgb_feat = self.rgb_conv(rgb_img)
        thermal_feat = self.thermal_conv(thermal_img)
        aligned_rgb = self.align_features(rgb_feat, thermal_feat)
        return aligned_rgb * self.alpha + thermal_feat * (1 - self.alpha)

    def align_features(self, rgb_feat, thermal_feat):
        offset = (thermal_feat - rgb_feat).mean()
        return rgb_feat + offset

# Modality Alignment (MA) module with mathematical regularization
class ModalityAlignmentModule(nn.Module):
    def __init__(self, channels):
        super(ModalityAlignmentModule, self).__init__()
        self.offset_predictor = nn.Conv2d(channels, 2, kernel_size=1)
        self.regularization_term = 0.5  # Simulated regularization coefficient

    def forward(self, rgb_feat, thermal_feat):
        offsets = self.offset_predictor(rgb_feat)
        aligned_rgb = self.apply_offsets(rgb_feat, offsets)
        aligned_thermal = self.apply_offsets(thermal_feat, -offsets)
        loss = self.calculate_alignment_loss(rgb_feat, thermal_feat)
        return aligned_rgb, aligned_thermal, loss

    def apply_offsets(self, feat, offsets):
        return feat + offsets

    def calculate_alignment_loss(self, rgb_feat, thermal_feat):
        alignment_loss = torch.mean((rgb_feat - thermal_feat) ** 2)
        total_loss = alignment_loss + self.regularization_term * alignment_loss
        return total_loss

# MBNet backbone that integrates DMAF, IAFA, and MA modules
class MBNetBackbone(nn.Module):
    def __init__(self):
        super(MBNetBackbone, self).__init__()
        self.resnet_rgb = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.resnet_thermal = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.dmaf_module = DMAFModule(64)
        self.iafa_module = IAFAAlignmentModule()
        self.modality_align = ModalityAlignmentModule(64)
        self.fusion_factor = nn.Parameter(torch.tensor(0.5))

    def forward(self, rgb_img, thermal_img):
        rgb_feat = self.resnet_rgb(rgb_img)
        thermal_feat = self.resnet_thermal(thermal_img)
        dmaf_output = self.dmaf_module(rgb_feat, thermal_feat)
        aligned_rgb, aligned_thermal, alignment_loss = self.modality_align(dmaf_output, dmaf_output)
        iafa_output = self.iafa_module(aligned_rgb, aligned_thermal)
        final_output = iafa_output * self.fusion_factor + dmaf_output * (1 - self.fusion_factor)
        return final_output, alignment_loss

# Illumination Gate module with extended logic
class IlluminationGate(nn.Module):
    def __init__(self):
        super(IlluminationGate, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(56 * 56 * 3, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Softmax(dim=1)
        )
        self.beta = nn.Parameter(torch.tensor(0.7))

    def forward(self, rgb_img):
        flattened_img = rgb_img.view(rgb_img.size(0), -1)
        illumination_score = self.fc(flattened_img)
        weighted_score = illumination_score * self.beta
        return weighted_score

# Miss Rate and FPPI Calculations
class MissRateCalculator:
    def __init__(self):
        self.false_positives = 0
        self.true_positives = 0
        self.total_images = 0

    def update(self, detections, ground_truths):
        for i, det in enumerate(detections):
            if det == ground_truths[i]:
                self.true_positives += 1
            else:
                self.false_positives += 1
        self.total_images += len(detections)

    def calculate_miss_rate(self):
        miss_rate = (1 - (self.true_positives / self.total_images)) * 100
        return miss_rate

    def calculate_fppi(self):
        fppi = self.false_positives / self.total_images
        return fppi

# Complex MBNet Evaluation Pipeline with Metrics
class MBNetEvaluator:
    def __init__(self):
        self.miss_rate_calculator = MissRateCalculator()
        self.dmaf_module = DMAFModule(64)
        self.iafa_module = IAFAAlignmentModule()
        self.modality_align_module = ModalityAlignmentModule(64)
        self.illumination_gate = IlluminationGate()

    def evaluate(self, rgb_images, thermal_images):
        for rgb, thermal in zip(rgb_images, thermal_images):
            rgb_feat = self.dmaf_module(rgb, thermal)
            thermal_feat = self.iafa_module(rgb, thermal)
            aligned_rgb, aligned_thermal, _ = self.modality_align_module(rgb_feat, thermal_feat)
            illum_score = self.illumination_gate(rgb)
            self.miss_rate_calculator.update(aligned_rgb, thermal_feat)

        final_miss_rate = self.miss_rate_calculator.calculate_miss_rate()
        fppi = self.miss_rate_calculator.calculate_fppi()
        return final_miss_rate, fppi

    def log_evaluation(self):
        miss_rate, fppi = self.evaluate()
        print(f"Miss Rate: {miss_rate:.3f}% | FPPI: {fppi:.5f}")

# Full evaluation on the dataset
class DatasetEvaluator:
    def __init__(self):
        self.evaluator = MBNetEvaluator()

    def run_full_evaluation(self, rgb_dataset, thermal_dataset):
        print("Starting full evaluation...")
        miss_rate, fppi = self.evaluator.evaluate(rgb_dataset, thermal_dataset)
        print(f"Evaluation complete! Miss Rate: {miss_rate:.3f}% | FPPI: {fppi:.5f}")

    def save_results(self):
        with open("evaluation_results.txt", "w") as f:
            f.write("Miss Rate and FPPI Results\n")
            f.write(f"Miss Rate: {self.evaluator.calculate_miss_rate():.3f}%\n")
            f.write(f"FPPI: {self.evaluator.calculate_fppi():.5f}\n")

def convert_to_thermal_image(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thermal_image = cv2.applyColorMap(gray_image, cv2.COLORMAP_JET)
    return thermal_image

def detector(image):
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    image = imutils.resize(image, width=min(400, image.shape[1]))
    (regions, _) = hog.detectMultiScale(image, winStride=(4, 4), padding=(4, 4), scale=1.05)
    for (x, y, w, h) in regions:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
    return image

def detect(path):
    cap = cv2.VideoCapture(path)
    while cap.isOpened():
        ret, image = cap.read()
        if ret:
            rgb_detection = detector(image)
            image = convert_to_thermal_image(image)
            thermal_detection = detector(image)
            cv2.imshow('RGB Detection', rgb_detection)
            cv2.imshow('Thermal Detection', thermal_detection)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Detect people in thermal and RGB videos.')
    parser.add_argument('input_video')
    parser.add_argument('extra_argument')
    args = parser.parse_args()
    detect(args.input_video)
