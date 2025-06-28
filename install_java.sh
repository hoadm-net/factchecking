#!/bin/bash

# Script ngắn gọn kiểm tra và cài Java trên Ubuntu
# Cài Java 21 LTS (phiên bản mới nhất LTS)

echo "🔍 Kiểm tra Java..."

if command -v java &> /dev/null; then
    JAVA_VERSION=$(java -version 2>&1 | head -n 1)
    echo "✅ Java đã cài đặt: $JAVA_VERSION"
    
    if [[ -z "$JAVA_HOME" ]]; then
        echo "⚠️  JAVA_HOME chưa thiết lập"
        JAVA_PATH=$(readlink -f $(which java))
        JAVA_HOME=$(dirname $(dirname $JAVA_PATH))
        echo "export JAVA_HOME=$JAVA_HOME" >> ~/.bashrc
        echo "✅ Đã thêm JAVA_HOME=$JAVA_HOME vào ~/.bashrc"
    else
        echo "✅ JAVA_HOME: $JAVA_HOME"
    fi
else
    echo "❌ Java chưa cài đặt. Đang cài Java 21..."
    
    # Cập nhật package list
    sudo apt update
    
    # Cài Java 21 JDK
    sudo apt install -y openjdk-21-jdk
    
    if command -v java &> /dev/null; then
        echo "✅ Cài đặt Java thành công!"
        
        # Thiết lập JAVA_HOME
        JAVA_PATH=$(readlink -f $(which java))
        JAVA_HOME=$(dirname $(dirname $JAVA_PATH))
        echo "export JAVA_HOME=$JAVA_HOME" >> ~/.bashrc
        echo "export PATH=\$JAVA_HOME/bin:\$PATH" >> ~/.bashrc
        
        echo "✅ JAVA_HOME đã thiết lập: $JAVA_HOME"
        echo "📝 Chạy 'source ~/.bashrc' để áp dụng cấu hình"
        
        # Hiển thị thông tin
        echo "----------------------------------------"
        java -version
        echo "----------------------------------------"
    else
        echo "❌ Cài đặt Java thất bại!"
        exit 1
    fi
fi

echo "🎉 Hoàn tất!" 
