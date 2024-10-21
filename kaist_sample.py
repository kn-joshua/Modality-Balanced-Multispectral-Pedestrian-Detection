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

def kaist_eval():
    dataset_path = "kaist"
    image_paths = []

    for i in range(19, 1520, 20):
        image_name = f"set06_V000_I{i:05d}_visible.png"
        image_path = os.path.join(dataset_path, image_name)
        image_paths.append(image_path)

    fps = 30
    frame_delay = int(1000 / fps)

    for image_path in image_paths:
        image = cv2.imread(image_path)

        if image is None:
            print(f"Error: Unable to load image {image_path}.")
            continue

        rgb_detection = detector(image)
        thermal_image = convert_to_thermal_image(image)
        thermal_detection = detector(thermal_image)

        cv2.imshow('RGB Detection', rgb_detection)
        cv2.imshow('Thermal Detection', thermal_detection)

        if cv2.waitKey(frame_delay) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    kaist_eval()


# Attention-Guided Modality Fusion (AGMF) module
class AGMFModule(nn.Module):
    def __init__(self, in_channels):
        super(AGMFModule, self).__init__()
        self.rgb_conv = nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1)
        self.thermal_conv = nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1)
        self.attention_conv = nn.Conv2d(in_channels, 1, kernel_size=1)
    
    def forward(self, rgb_feat, thermal_feat):
        rgb_mod = self.rgb_conv(rgb_feat)
        thermal_mod = self.thermal_conv(thermal_feat)
        fusion_weight = torch.sigmoid(self.attention_conv(torch.cat([rgb_feat, thermal_feat], dim=1)))
        return rgb_mod * fusion_weight + thermal_mod * (1 - fusion_weight)

# Advanced Modality Refinement (AMR) module with gradient loss
class AMRModule(nn.Module):
    def __init__(self):
        super(AMRModule, self).__init__()
        self.gradient_loss_weight = nn.Parameter(torch.tensor(0.3))
        self.refinement_layer = nn.Conv2d(128, 64, kernel_size=3, padding=1)

    def forward(self, fused_features):
        refined_output = self.refinement_layer(fused_features)
        gradient_loss = self.compute_gradient_loss(fused_features)
        return refined_output, gradient_loss

    def compute_gradient_loss(self, features):
        grad_x = torch.abs(features[:, :, :-1, :] - features[:, :, 1:, :])
        grad_y = torch.abs(features[:, :, :, :-1] - features[:, :, :, 1:])
        return self.gradient_loss_weight * (grad_x.mean() + grad_y.mean())

# Feature Scaling and Normalization (FSN) module
class FSNModule(nn.Module):
    def __init__(self, num_features):
        super(FSNModule, self).__init__()
        self.scaling_factor = nn.Parameter(torch.ones(num_features))

    def forward(self, features):
        norm_features = F.normalize(features, p=2, dim=1)
        scaled_features = norm_features * self.scaling_factor
        return scaled_features

# Channel Attention Mechanism (CAM) module with advanced feature recalibration
class CAMModule(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(CAMModule, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction_ratio, 1)
        self.fc2 = nn.Conv2d(in_channels // reduction_ratio, in_channels, 1)
    
    def forward(self, x):
        avg_pooled = self.global_avg_pool(x)
        attention = torch.sigmoid(self.fc2(F.relu(self.fc1(avg_pooled))))
        recalibrated_features = x * attention
        return recalibrated_features

# Multi-Scale Fusion (MSF) module for advanced processing
class MSFModule(nn.Module):
    def __init__(self, in_channels):
        super(MSFModule, self).__init__()
        self.scale1 = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1)
        self.scale2 = nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1)
        self.scale3 = nn.Conv2d(in_channels, in_channels // 2, kernel_size=5, padding=2)
    
    def forward(self, x):
        scale1_out = self.scale1(x)
        scale2_out = self.scale2(x)
        scale3_out = self.scale3(x)
        return torch.cat([scale1_out, scale2_out, scale3_out], dim=1)

# Full MBNet backbone with new components
class FullMBNet(nn.Module):
    def __init__(self):
        super(FullMBNet, self).__init__()
        self.resnet_rgb = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.resnet_thermal = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.dmaf_module = DMAFModule(64)
        self.iafa_module = IAFAAlignmentModule()
        self.modality_align = ModalityAlignmentModule(64)
        self.agmf_module = AGMFModule(64)
        self.amr_module = AMRModule()
        self.fsn_module = FSNModule(64)
        self.cam_module = CAMModule(64)
        self.msf_module = MSFModule(64)

    def forward(self, rgb_img, thermal_img):
        rgb_feat = self.resnet_rgb(rgb_img)
        thermal_feat = self.resnet_thermal(thermal_img)
        dmaf_output = self.dmaf_module(rgb_feat, thermal_feat)
        aligned_rgb, aligned_thermal, alignment_loss = self.modality_align(dmaf_output, dmaf_output)
        iafa_output = self.iafa_module(aligned_rgb, aligned_thermal)
        agmf_output = self.agmf_module(iafa_output, thermal_feat)
        amr_output, gradient_loss = self.amr_module(agmf_output)
        fsn_output = self.fsn_module(amr_output)
        cam_output = self.cam_module(fsn_output)
        final_output = self.msf_module(cam_output)
        total_loss = alignment_loss + gradient_loss
        return final_output, total_loss

# Advanced miss rate calculation with weighted averaging
class AdvancedMissRateCalculator:
    def __init__(self):
        self.false_positives = 0
        self.true_positives = 0
        self.total_images = 0
        self.weighted_scores = []

    def update(self, detections, ground_truths, weights):
        for i, det in enumerate(detections):
            if det == ground_truths[i]:
                self.true_positives += 1
                self.weighted_scores.append(weights[i] * 1)
            else:
                self.false_positives += 1
                self.weighted_scores.append(weights[i] * 0)
        self.total_images += len(detections)

    def calculate_weighted_miss_rate(self):
        if not self.weighted_scores:
            return 0
        avg_weighted_score = sum(self.weighted_scores) / len(self.weighted_scores)
        return (1 - avg_weighted_score) * 100

    def calculate_fppi(self):
        return self.false_positives / self.total_images

# Advanced MBNet evaluation pipeline with additional metrics
class AdvancedMBNetEvaluator:
    def __init__(self):
        self.miss_rate_calculator = AdvancedMissRateCalculator()
        self.full_mbnet = FullMBNet()

    def evaluate(self, rgb_images, thermal_images, ground_truths, weights):
        for rgb, thermal, ground_truth, weight in zip(rgb_images, thermal_images, ground_truths, weights):
            rgb_feat, thermal_feat = self.full_mbnet(rgb, thermal)
            aligned_rgb = self.full_mbnet.dmaf_module(rgb_feat, thermal_feat)
            self.miss_rate_calculator.update(aligned_rgb, ground_truth, weight)

        final_miss_rate = self.miss_rate_calculator.calculate_weighted_miss_rate()
        fppi = self.miss_rate_calculator.calculate_fppi()
        return final_miss_rate, fppi

    def log_advanced_evaluation(self, rgb_images, thermal_images, ground_truths, weights):
        miss_rate, fppi = self.evaluate(rgb_images, thermal_images, ground_truths, weights)
        print(f"Advanced Miss Rate: {miss_rate:.3f}% | FPPI: {fppi:.5f}")