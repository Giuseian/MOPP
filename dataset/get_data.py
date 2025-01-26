import os

# Install required system dependencies
os.system("""
apt-get install -y \
   libgl1-mesa-dev \
   libgl1-mesa-glx \
   libglew-dev \
   libosmesa6-dev \
   software-properties-common
apt-get install -y patchelf
""")

# Install Python dependencies
os.system("pip install free-mujoco-py")
os.system("pip install git+https://github.com/Farama-Foundation/d4rl@master#egg=d4rl")
os.system("pip install gymnasium minari carla")

print("Dependencies and datasets have been installed successfully!")

# Setup Mujoco
if not os.path.exists('.mujoco_setup_complete'):
    os.system("""
    apt-get -qq update
    apt-get -qq install -y libosmesa6-dev libgl1-mesa-glx libglfw3 libgl1-mesa-dev libglew-dev patchelf
    mkdir ~/.mujoco
    wget -q https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz -O mujoco.tar.gz
    tar -zxf mujoco.tar.gz -C "$HOME/.mujoco"
    rm mujoco.tar.gz
    echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin' >> ~/.bashrc 
    echo 'export LD_PRELOAD=$LD_PRELOAD:/usr/lib/x86_64-linux-gnu/libGLEW.so' >> ~/.bashrc 
    echo "/root/.mujoco/mujoco210/bin" > /etc/ld.so.conf.d/mujoco_ld_lib_path.conf
    ldconfig
    pip3 install -U 'mujoco-py<2.2,>=2.1'
    touch .mujoco_setup_complete
    """)
    print("Mujoco environment setup complete!")

# Set environment variables
os.environ['LD_LIBRARY_PATH'] = os.environ.get('LD_LIBRARY_PATH', '') + ':/root/.mujoco/mujoco210/bin'
os.environ['LD_PRELOAD'] = os.environ.get('LD_PRELOAD', '') + ':/usr/lib/x86_64-linux-gnu/libGLEW.so'