from PIL import Image
import os
from pathlib import Path
import sys

def convert_to_pdf(input_path: str, output_dir: str = None):
    """Convert a JPEG image to PDF"""
    try:
        # Open the image
        print(f"[DEBUG] Opening image: {input_path}")
        image = Image.open(input_path)
        
        # Convert to RGB if necessary (PDF doesn't support RGBA)
        if image.mode == 'RGBA':
            print("[DEBUG] Converting RGBA to RGB")
            image = image.convert('RGB')
        
        # Determine output path
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, Path(input_path).stem + '.pdf')
        else:
            output_path = str(Path(input_path).with_suffix('.pdf'))
        
        # Save as PDF
        print(f"[DEBUG] Saving PDF to: {output_path}")
        image.save(output_path, 'PDF', resolution=100.0)
        print(f"[SUCCESS] Converted {input_path} to PDF")
        return output_path
        
    except Exception as e:
        print(f"[ERROR] Failed to convert {input_path}: {str(e)}")
        return None

def convert_directory(input_dir: str, output_dir: str = None):
    """Convert all JPEG/JPG files in a directory to PDF"""
    if not os.path.exists(input_dir):
        print(f"[ERROR] Input directory does not exist: {input_dir}")
        return
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    print(f"[DEBUG] Scanning directory: {input_dir}")
    converted = 0
    failed = 0
    
    for file in os.listdir(input_dir):
        if file.lower().endswith(('.jpg', '.jpeg')):
            input_path = os.path.join(input_dir, file)
            result = convert_to_pdf(input_path, output_dir)
            if result:
                converted += 1
            else:
                failed += 1
    
    print(f"\n[SUMMARY] Converted: {converted}, Failed: {failed}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Single file: python convert_to_pdf.py input.jpg [output_dir]")
        print("  Directory:   python convert_to_pdf.py --dir input_dir [output_dir]")
        sys.exit(1)
    
    if sys.argv[1] == "--dir":
        if len(sys.argv) < 3:
            print("[ERROR] Please provide input directory")
            sys.exit(1)
        input_dir = sys.argv[2]
        output_dir = sys.argv[3] if len(sys.argv) > 3 else None
        convert_directory(input_dir, output_dir)
    else:
        input_path = sys.argv[1]
        output_dir = sys.argv[2] if len(sys.argv) > 2 else None
        convert_to_pdf(input_path, output_dir) 