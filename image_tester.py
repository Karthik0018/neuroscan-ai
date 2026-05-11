import os
import numpy as np
import matplotlib.pyplot as plt
import shutil
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from collections import defaultdict

# Configuration
BASE_DIR = r"C:\Users\karth\Desktop\ML with Python\brain-tumor-class\brain-tumor"
TEST_DIR = os.path.join(BASE_DIR, "Testing")
OUTPUT_DIR = os.path.join(BASE_DIR, "vgg16_analysis")
MODEL_PATH = os.path.join(BASE_DIR, "saved_models", "VGG16.h5")

IMG_SIZE = (128, 128)
CLASSES = ['glioma', 'meningioma', 'notumor', 'pituitary']
CLASS_NAMES = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

# Create output directory structure
def create_output_structure():
    """Create folders for each tumor type with subfolders for predictions"""
    for tumor_type in CLASSES:
        # Main folder for each tumor type
        tumor_folder = os.path.join(OUTPUT_DIR, f"{tumor_type}_analysis")
        
        # Subfolders for each prediction outcome
        for pred_type in CLASSES:
            subfolder = os.path.join(tumor_folder, f"predicted_as_{pred_type}")
            os.makedirs(subfolder, exist_ok=True)
        
        # Create a summary folder for visualizations
        summary_folder = os.path.join(tumor_folder, "summary_plots")
        os.makedirs(summary_folder, exist_ok=True)
    
    # Create overall summary folder
    overall_summary = os.path.join(OUTPUT_DIR, "overall_summary")
    os.makedirs(overall_summary, exist_ok=True)
    
    print(f"✓ Output directory structure created at: {OUTPUT_DIR}")

def load_and_prepare_model():
    """Load the VGG16 model"""
    print(f"\n📁 Loading model from: {MODEL_PATH}")
    model = load_model(MODEL_PATH)
    print(f"✓ Model loaded successfully")
    print(f"  Input shape: {model.input_shape}")
    print(f"  Output shape: {model.output_shape}")
    return model

def predict_and_categorize(model):
    """Go through test dataset and categorize images by true label and predictions"""
    
    # Data generator for loading images without augmentation
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    # Statistics dictionary
    stats = defaultdict(lambda: defaultdict(int))
    misclassified_summary = defaultdict(list)
    
    print(f"\n🔍 Analyzing test dataset...")
    print("="*60)
    
    # Process each tumor type separately
    for tumor_type in CLASSES:
        tumor_path = os.path.join(TEST_DIR, tumor_type)
        if not os.path.exists(tumor_path):
            print(f"⚠️ Warning: {tumor_path} not found")
            continue
        
        images = [f for f in os.listdir(tumor_path) if f.endswith(('.jpg', '.png', '.jpeg'))]
        print(f"\n📊 Processing {tumor_type.upper()} - {len(images)} images")
        
        for img_name in images:
            img_path = os.path.join(tumor_path, img_name)
            
            # Load and preprocess image
            img = Image.open(img_path).convert('RGB').resize(IMG_SIZE)
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            # Predict
            predictions = model.predict(img_array, verbose=0)
            pred_idx = np.argmax(predictions[0])
            pred_class = CLASSES[pred_idx]
            confidence = predictions[0][pred_idx] * 100
            
            # Get true class index
            true_idx = CLASSES.index(tumor_type)
            is_correct = (pred_class == tumor_type)
            
            # Update statistics
            stats[tumor_type][pred_class] += 1
            
            # Copy image to appropriate folder
            dest_folder = os.path.join(OUTPUT_DIR, f"{tumor_type}_analysis", f"predicted_as_{pred_class}")
            dest_path = os.path.join(dest_folder, f"{img_name}")
            shutil.copy2(img_path, dest_path)
            
            # Store misclassified info for summary
            if not is_correct:
                misclassified_summary[tumor_type].append({
                    'filename': img_name,
                    'predicted': pred_class,
                    'confidence': confidence,
                    'path': dest_path
                })
    
    return stats, misclassified_summary

def create_confusion_matrix_plot(stats):
    """Create confusion matrix visualization"""
    
    # Create confusion matrix
    cm = np.zeros((len(CLASSES), len(CLASSES)))
    for i, true_class in enumerate(CLASSES):
        for j, pred_class in enumerate(CLASSES):
            cm[i][j] = stats[true_class][pred_class]
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation='nearest', cmap='YlOrRd')
    ax.set_xticks(np.arange(len(CLASSES)))
    ax.set_yticks(np.arange(len(CLASSES)))
    ax.set_xticklabels([c.upper() for c in CLASSES], rotation=45, ha='right')
    ax.set_yticklabels([c.upper() for c in CLASSES])
    
    # Add text annotations
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            text = ax.text(j, i, int(cm[i, j]),
                          ha="center", va="center", color="white" if cm[i, j] > cm.max()/2 else "black",
                          fontweight='bold')
    
    ax.set_xlabel('Predicted Class', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Class', fontsize=12, fontweight='bold')
    ax.set_title('VGG16 - Confusion Matrix (Test Set)', fontsize=14, fontweight='bold')
    
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'overall_summary', 'confusion_matrix.png'), dpi=150, bbox_inches='tight')
    plt.show()
    
    return cm

def create_tumor_type_summary_plots(stats, misclassified_summary):
    """Create detailed plots for each tumor type"""
    
    for tumor_type in CLASSES:
        tumor_folder = os.path.join(OUTPUT_DIR, f"{tumor_type}_analysis", "summary_plots")
        
        # Plot 1: Prediction distribution for this tumor type
        predictions = stats[tumor_type]
        pred_classes = list(predictions.keys())
        counts = list(predictions.values())
        colors = ['#2ECC71' if p == tumor_type else '#E74C3C' for p in pred_classes]
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Bar chart
        bars = axes[0].bar(pred_classes, counts, color=colors, edgecolor='black', linewidth=1.5)
        axes[0].set_title(f'{tumor_type.upper()} - Prediction Distribution', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Predicted Class', fontsize=10)
        axes[0].set_ylabel('Number of Images', fontsize=10)
        axes[0].tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, count in zip(bars, counts):
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                        str(count), ha='center', va='bottom', fontweight='bold')
        
        # Pie chart
        correct_count = predictions[tumor_type]
        incorrect_count = sum(counts) - correct_count
        sizes = [correct_count, incorrect_count]
        labels = [f'Correct\n({correct_count})', f'Incorrect\n({incorrect_count})']
        colors_pie = ['#2ECC71', '#E74C3C']
        axes[1].pie(sizes, labels=labels, colors=colors_pie, autopct='%1.1f%%', startangle=90,
                   textprops={'fontweight': 'bold'})
        axes[1].set_title(f'{tumor_type.upper()} - Accuracy: {correct_count/sum(counts)*100:.1f}%',
                         fontsize=12, fontweight='bold')
        
        plt.suptitle(f'Analysis for {tumor_type.upper()} Tumor', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(tumor_folder, f'{tumor_type}_analysis.png'), dpi=150, bbox_inches='tight')
        plt.show()
        
        # Create misclassified gallery if there are misclassifications
        if tumor_type in misclassified_summary and misclassified_summary[tumor_type]:
            create_misclassified_gallery(tumor_type, misclassified_summary[tumor_type], tumor_folder)

def create_misclassified_gallery(tumor_type, misclassified_images, output_folder):
    """Create a gallery of misclassified images"""
    
    if not misclassified_images:
        return
    
    # Determine grid size
    n_images = min(len(misclassified_images), 12)  # Max 12 images per gallery
    n_cols = 4
    n_rows = (n_images + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for idx, mis_info in enumerate(misclassified_images[:n_images]):
        row = idx // n_cols
        col = idx % n_cols
        
        # Load and display image
        img = Image.open(os.path.join(TEST_DIR, tumor_type, mis_info['filename']))
        axes[row, col].imshow(img)
        axes[row, col].axis('off')
        
        # Add title with prediction info
        pred_class_upper = mis_info['predicted'].upper()
        confidence = mis_info['confidence']
        axes[row, col].set_title(f'True: {tumor_type.upper()}\nPred: {pred_class_upper}\nConf: {confidence:.1f}%',
                                color='red', fontsize=9, fontweight='bold')
    
    # Hide unused subplots
    for idx in range(len(misclassified_images), n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].axis('off')
    
    plt.suptitle(f'{tumor_type.upper()} - Misclassified Images Gallery', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f'{tumor_type}_misclassified_gallery.png'), dpi=150, bbox_inches='tight')
    plt.show()

def create_overall_report(stats, misclassified_summary, cm):
    """Create comprehensive text report"""
    
    report_path = os.path.join(OUTPUT_DIR, 'overall_summary', 'analysis_report.txt')
    
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("VGG16 BRAIN TUMOR CLASSIFICATION - DETAILED ANALYSIS REPORT\n")
        f.write("="*80 + "\n\n")
        
        # Overall statistics
        total_correct = 0
        total_images = 0
        
        for true_class in CLASSES:
            class_total = sum(stats[true_class].values())
            class_correct = stats[true_class][true_class]
            class_accuracy = (class_correct / class_total) * 100
            total_correct += class_correct
            total_images += class_total
            
            f.write(f"\n{true_class.upper()} TUMOR ANALYSIS:\n")
            f.write("-"*50 + "\n")
            f.write(f"Total images: {class_total}\n")
            f.write(f"Correctly classified: {class_correct}\n")
            f.write(f"Accuracy: {class_accuracy:.2f}%\n")
            f.write(f"Misclassified: {class_total - class_correct}\n\n")
            f.write("Prediction breakdown:\n")
            
            for pred_class, count in stats[true_class].items():
                status = "✓ CORRECT" if pred_class == true_class else "✗ INCORRECT"
                f.write(f"  → Predicted as {pred_class.upper():12s}: {count:3d} images ({status})\n")
            
            # List misclassified sample filenames
            if true_class in misclassified_summary and misclassified_summary[true_class]:
                f.write(f"\n  Misclassified image samples:\n")
                for mis_info in misclassified_summary[true_class][:5]:
                    f.write(f"    - {mis_info['filename']:<40} (pred: {mis_info['predicted']}, conf: {mis_info['confidence']:.1f}%)\n")
        
        f.write("\n" + "="*50 + "\n")
        f.write("OVERALL PERFORMANCE\n")
        f.write("="*50 + "\n")
        f.write(f"Total images analyzed: {total_images}\n")
        f.write(f"Overall accuracy: {(total_correct/total_images)*100:.2f}%\n")
        f.write(f"Total misclassifications: {total_images - total_correct}\n\n")
        
        f.write("\nCONFUSION MATRIX:\n")
        f.write("-"*50 + "\n")
        f.write("True\\Pred    ")
        for cls in CLASSES:
            f.write(f"{cls[:4].upper():>6} ")
        f.write("\n")
        
        for i, true_class in enumerate(CLASSES):
            f.write(f"{true_class[:8].upper():<12} ")
            for j, pred_class in enumerate(CLASSES):
                f.write(f"{int(cm[i, j]):>6} ")
            f.write("\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("END OF REPORT\n")
        f.write("="*80 + "\n")
    
    print(f"\n✓ Detailed report saved to: {report_path}")

def analyze_image_orientations():
    """Helper function to tag images that might have orientation issues"""
    
    print("\n📐 Analyzing potential orientation issues...")
    print("="*60)
    print("Note: This analysis is based on visual inspection of misclassified images.")
    print("You can manually review the misclassified images in the output folders to")
    print("identify patterns such as:")
    print("  • Different scanning angles (top, side, angled views)")
    print("  • Image quality issues (blur, artifacts, noise)")
    print("  • Unusual tumor presentations")
    print("  • Different MRI sequences (T1, T2, FLAIR, etc.)")
    print("\nThe misclassified images have been organized in:")
    print(f"  {OUTPUT_DIR}")
    print("\nFor each tumor type, check the 'predicted_as_*' folders to see")
    print("what the model confused them with.\n")

def main():
    """Main execution function"""
    
    print("\n" + "🚀"*30)
    print("VGG16 BRAIN TUMOR CLASSIFICATION - MISCLASSIFICATION ANALYZER")
    print("🚀"*30)
    
    # Step 1: Create output structure
    create_output_structure()
    
    # Step 2: Load model
    model = load_and_prepare_model()
    
    # Step 3: Predict and categorize
    stats, misclassified_summary = predict_and_categorize(model)
    
    # Step 4: Create visualizations
    print("\n📊 Creating visualizations...")
    cm = create_confusion_matrix_plot(stats)
    
    for tumor_type in CLASSES:
        create_tumor_type_summary_plots(stats, misclassified_summary)
    
    # Step 5: Generate report
    create_overall_report(stats, misclassified_summary, cm)
    
    # Step 6: Provide analysis guidance
    analyze_image_orientations()
    
    print("\n✅ ANALYSIS COMPLETE!")
    print(f"\n📁 All results saved to: {OUTPUT_DIR}")
    print("\nFOLDER STRUCTURE CREATED:")
    print("  vgg16_analysis/")
    print("  ├── glioma_analysis/")
    print("  │   ├── predicted_as_glioma/     (Correct predictions)")
    print("  │   ├── predicted_as_meningioma/  (Misclassified as meningioma)")
    print("  │   ├── predicted_as_notumor/     (Misclassified as no tumor)")
    print("  │   ├── predicted_as_pituitary/   (Misclassified as pituitary)")
    print("  │   └── summary_plots/            (Visualizations)")
    print("  ├── meningioma_analysis/")
    print("  ├── notumor_analysis/")
    print("  ├── pituitary_analysis/")
    print("  └── overall_summary/              (Report & confusion matrix)")
    
    # Print quick statistics
    print("\n📊 QUICK STATISTICS:")
    print("-"*40)
    for tumor_type in CLASSES:
        correct = stats[tumor_type][tumor_type]
        total = sum(stats[tumor_type].values())
        acc = (correct/total)*100
        print(f"{tumor_type.upper():12s}: {correct:3d}/{total:3d} ({acc:.1f}%)")
    
    total_correct = sum(stats[t][t] for t in CLASSES)
    total_images = sum(sum(stats[t].values()) for t in CLASSES)
    print("-"*40)
    print(f"{'OVERALL':12s}: {total_correct:3d}/{total_images:3d} ({(total_correct/total_images)*100:.1f}%)")
    print("-"*40)

if __name__ == "__main__":
    main()