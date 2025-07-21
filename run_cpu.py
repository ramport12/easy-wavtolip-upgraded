#!/usr/bin/env python3
"""
Easy-Wav2Lip CPU Launcher
Simplified launcher for CPU-only inference without advanced features
"""

import os
import sys
import argparse
from pathlib import Path

def main():
    print("🎬 Easy-Wav2Lip CPU Edition")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not os.path.exists('inference.py'):
        print("❌ Error: inference.py not found. Please run from the Easy-Wav2Lip directory.")
        return 1
    
    # Check for required model files
    if not os.path.exists('checkpoints/mobilenet.pth'):
        print("❌ Error: mobilenet.pth not found in checkpoints/")
        print("Please download the required model files first.")
        return 1
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Easy-Wav2Lip CPU Inference")
    parser.add_argument("--video", "-v", required=True, help="Path to input video file")
    parser.add_argument("--audio", "-a", help="Path to input audio file (optional)")
    parser.add_argument("--output", "-o", help="Output file path (default: output.mp4)")
    parser.add_argument("--quality", choices=["fast", "improved", "enhanced"], default="improved",
                       help="Processing quality (default: improved)")
    parser.add_argument("--height", type=int, default=480, 
                       help="Output height (default: 480)")
    parser.add_argument("--preview", action="store_true", 
                       help="Show preview frames during processing")
    parser.add_argument("--batch-size", type=int, default=1,
                       help="Batch size for CPU processing (default: 1)")
    parser.add_argument("--smooth", action="store_true", default=True,
                       help="Enable temporal smoothing (default: True)")
    parser.add_argument("--padding", nargs=4, type=int, metavar=('U', 'D', 'L', 'R'),
                       help="Custom padding: Up Down Left Right (e.g., 0 15 0 0)")
    
    args = parser.parse_args()
    
    # Validate input files
    if not os.path.exists(args.video):
        print(f"❌ Error: Video file not found: {args.video}")
        return 1
    
    if args.audio and not os.path.exists(args.audio):
        print(f"❌ Error: Audio file not found: {args.audio}")
        return 1
    
    # Set default output path
    if not args.output:
        video_path = Path(args.video)
        args.output = str(video_path.parent / f"{video_path.stem}_lipsynced.mp4")
    
    # Create inference arguments with CPU optimizations
    inference_args = [
        sys.executable, "inference.py",
        "--face", args.video,
        "--checkpoint_path", "checkpoints/wav2lip_gan.pth" if os.path.exists("checkpoints/wav2lip_gan.pth") else "checkpoints/wav2lip.pth",
        "--outfile", args.output,
        "--out_height", str(args.height),
        "--no_sr",  # Disable super resolution for CPU
        "--wav2lip_batch_size", str(args.batch_size),  # CPU-optimized batch size
        "--nosmooth", "False" if args.smooth else "True",  # Smoothing control
    ]
    
    # Add audio if provided
    if args.audio:
        inference_args.extend(["--audio", args.audio])
    else:
        inference_args.extend(["--audio", args.video])
    
    # Set quality-specific options with advanced feathering
    if args.quality == "enhanced":
        inference_args.extend([
            "--mask_feathering", "201",     # Maximum feathering for best quality
            "--mask_dilation", "2.2",       # Larger mask for enhanced blending
            "--quality", "Improved",        # Enable advanced mask blending
            "--mouth_tracking", "True",     # Enable mouth tracking
            "--mouth_tracking_backend", "mediapipe",  # Use MediaPipe for better tracking
            "--nosmooth", "False",          # Enable temporal smoothing
        ])
    elif args.quality == "improved":
        inference_args.extend([
            "--mask_feathering", "151",     # Good feathering value (must be odd)
            "--mask_dilation", "1.8",       # Improved mask size
            "--quality", "Improved",        # Enable mask blending
            "--mouth_tracking", "True",     # Enable mouth tracking for better alignment
            "--mouth_tracking_backend", "dlib",  # Use dlib for stability
        ])
    else:  # fast mode
        inference_args.extend([
            "--mask_feathering", "75",      # Light feathering for fast mode (odd number)
            "--mask_dilation", "1.4",       # Balanced mask size
            "--quality", "Fast",            # Fast processing
        ])
    
    # Apply custom padding if specified, otherwise use quality defaults
    if args.padding:
        inference_args.extend(["--pads"] + [str(p) for p in args.padding])
    else:
        # Default padding based on quality
        if args.quality == "enhanced":
            inference_args.extend(["--pads", "0", "15", "0", "0"])
        elif args.quality == "improved":
            inference_args.extend(["--pads", "0", "12", "0", "0"])
        else:
            inference_args.extend(["--pads", "0", "10", "0", "0"])
    
    # Enable preview if requested
    if args.preview:
        inference_args.append("--preview_input")
    
    print("🚀 Starting CPU inference...")
    print(f"📹 Input video: {args.video}")
    print(f"🎵 Audio source: {args.audio if args.audio else 'from video'}")
    print(f"📁 Output: {args.output}")
    print(f"⚙️  Quality: {args.quality}")
    print(f"📐 Height: {args.height}px")
    print("=" * 50)
    
    # Run inference
    try:
        import subprocess
        result = subprocess.run(inference_args, check=False)
        
        if result.returncode == 0:
            print("\n✅ Processing completed successfully!")
            print(f"📁 Output saved to: {args.output}")
            
            # Show file size
            if os.path.exists(args.output):
                size_mb = os.path.getsize(args.output) / (1024 * 1024)
                print(f"📊 File size: {size_mb:.1f} MB")
            
            return 0
        else:
            print("\n❌ Processing failed!")
            return result.returncode
            
    except KeyboardInterrupt:
        print("\n⚠️  Processing interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())