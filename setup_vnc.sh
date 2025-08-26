#!/bin/bash

# VNC Setup Script for Isaac Gym Visualization
# This script sets up a VNC server for remote visualization

echo "Setting up VNC server for Isaac Gym visualization..."

# Install VNC server if not present
if ! command -v vncserver &> /dev/null; then
    echo "Installing VNC server..."
    sudo apt update
    sudo apt install -y tigervnc-standalone-server tigervnc-common
fi

# Install desktop environment if needed
if ! dpkg -l | grep -q xfce4; then
    echo "Installing lightweight desktop environment..."
    sudo apt install -y xfce4 xfce4-goodies
fi

# Create VNC startup script
mkdir -p ~/.vnc
cat > ~/.vnc/xstartup << 'EOF'
#!/bin/bash
unset SESSION_MANAGER
unset DBUS_SESSION_BUS_ADDRESS
export XKL_XMODMAP_DISABLE=1
export XDG_CURRENT_DESKTOP="XFCE"
export XDG_MENU_PREFIX="xfce-"

# Set up environment for Isaac Gym
export MESA_GL_VERSION_OVERRIDE=3.3
export MESA_GLSL_VERSION_OVERRIDE=330

# Start XFCE desktop
startxfce4 &
EOF

chmod +x ~/.vnc/xstartup

# Set VNC password if not set
if [ ! -f ~/.vnc/passwd ]; then
    echo "Setting VNC password..."
    echo "Please set a password for VNC access:"
    vncpasswd
fi

# Start VNC server
echo "Starting VNC server on display :1..."
vncserver :1 -geometry 1920x1080 -depth 24

echo ""
echo "VNC server started successfully!"
echo "Connection details:"
echo "  - VNC Display: :1"
echo "  - Port: 5901"
echo "  - Resolution: 1920x1080"
echo ""
echo "To connect:"
echo "  1. Use SSH tunnel: ssh -L 5901:localhost:5901 user@your-server"
echo "  2. Connect VNC client to localhost:5901"
echo ""
echo "To stop VNC server: vncserver -kill :1"
echo ""
echo "Now you can run Isaac Gym with visualization:"
echo "  export DISPLAY=:1"
echo "  ./run_with_display.sh train g1" 