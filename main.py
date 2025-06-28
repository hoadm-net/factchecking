#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MINT TextGraph - Main Demo
Simple CLI wrapper and usage demonstration
"""

import sys
import os
from mint.cli import main as cli_main

def show_help():
    """Show usage examples and help"""
    print("""
🔥 MINT TextGraph - Vietnamese Text Graph Analysis
==================================================

Available Commands:
  mint-graph  - Main CLI command
  textgraph   - Alias for mint-graph
  python main.py (no args) - Run demo with data/demo.json

Quick Examples:
  
1. Demo với dữ liệu từ data/demo.json:
   python main.py
   mint-graph --demo --verbose

2. Phân tích text tùy chỉnh:
   mint-graph --context "Văn bản context..." --claim "Văn bản claim..." --export-image graph.png

3. Từ file JSON:
   mint-graph --input-file data.json --similarity-threshold 0.7 --top-k 3

4. Tắt tính năng:
   mint-graph --demo --disable-entities --disable-semantic

5. Tùy chỉnh tham số:
   mint-graph --demo --similarity-threshold 0.8 --top-k 5 --openai-model gpt-4o

6. Lưu nhiều format:
   mint-graph --demo --export-image graph.png --export-json data.json --export-graph graph.gexf

7. Chế độ yên lặng:
   mint-graph --demo --quiet --disable-visualization

Full Help:
  mint-graph --help

Các tham số chính:
  -c, --context         : Văn bản context
  -l, --claim           : Văn bản claim
  -f, --input-file      : File JSON input
  -d, --demo            : Chạy với dữ liệu demo
  -st, --similarity-threshold : Ngưỡng similarity (0.0-1.0)
  -k, --top-k           : Số từ tương tự kết nối
  -om, --openai-model   : Model OpenAI (gpt-4o-mini, gpt-4o, ...)
  -ei, --export-image   : Xuất ảnh đồ thị
  -eg, --export-graph   : Xuất file đồ thị
  -ej, --export-json    : Xuất JSON
  -v, --verbose         : Chi tiết
  -q, --quiet           : Yên lặng
  --disable-entities    : Tắt trích xuất entities
  --disable-semantic    : Tắt semantic similarity

Note: Đảm bảo đã cài đặt dependencies và có file .env với OPENAI_KEY
""")

def main():
    """Main entry point"""
    # If no arguments, run demo with data from data/demo.json
    if len(sys.argv) == 1:
        print("🚀 Running MINT TextGraph with demo data from data/demo.json...")
        print("📝 Use --help to see all available options\n")
        sys.argv.append('--demo')
        sys.argv.append('--verbose')
    
    # If first argument is help or --help, show help
    elif len(sys.argv) == 2 and sys.argv[1] in ['help', '--help', '-h']:
        show_help()
        return
    
    # Auto-add demo if no input source is specified
    else:
        has_input_source = any(arg in sys.argv for arg in ['--demo', '-d', '--context', '-c', '--input-file', '-f'])
        if not has_input_source:
            print("🚀 No input source detected, using demo data from data/demo.json...")
            print("📝 Use --help to see all available options\n")
            sys.argv.append('--demo')
    
    # Otherwise, delegate to CLI
    try:
        cli_main()
    except KeyboardInterrupt:
        print("\n❌ Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
    