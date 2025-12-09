#!/bin/bash
# ==============================================================================
# CHIMERA-VX SETUP SCRIPT
# The Ultimate CTF Installation
# ==============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Logging functions
log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_step() { echo -e "${PURPLE}[STEP]${NC} $1"; }
log_chimera() { echo -e "${CYAN}[CHIMERA]${NC} $1"; }

# Banner
print_banner() {
    clear
    echo -e "${CYAN}"
    echo "=================================================================================="
    echo "                       ░█████╗░██╗░░██╗██╗███╗░░░███╗███████╗██████╗░░█████╗░"
    echo "                       ██╔══██╗██║░░██║██║████╗░████║██╔════╝██╔══██╗██╔══██╗"
    echo "                       ██║░░╚═╝███████║██║██╔████╔██║█████╗░░██████╔╝███████║"
    echo "                       ██║░░██╗██╔══██║██║██║╚██╔╝██║██╔══╝░░██╔══██╗██╔══██║"
    echo "                       ╚█████╔╝██║░░██║██║██║░╚═╝░██║███████╗██║░░██║██║░░██║"
    echo "                       ░╚════╝░╚═╝░░╚═╝╚═╝╚═╝░░░░░╚═╝╚══════╝╚═╝░░╚═╝╚═╝░░╚═╝"
    echo "                                    V X   -   T H E   U L T I M A T E   C T F"
    echo "=================================================================================="
    echo -e "${NC}"
}

# Check if running as root
check_root() {
    if [[ $EUID -eq 0 ]]; then
        log_warning "Running as root is not recommended for Chimera-VX."
        read -p "Continue anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log_error "Please run as regular user."
            exit 1
        fi
    fi
}

# Detect platform
detect_platform() {
    log_step "Detecting platform..."
    if [[ -f /data/data/com.termux/files/usr/bin/termux-info ]]; then
        PLATFORM="termux"
        log_info "Platform: Termux (Android)"
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        PLATFORM="linux"
        log_info "Platform: Linux"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        PLATFORM="macos"
        log_info "Platform: macOS"
    else
        log_error "Unsupported platform: $OSTYPE"
        exit 1
    fi
}

# Update system packages
update_system() {
    log_step "Updating system packages..."
    
    if [[ "$PLATFORM" == "termux" ]]; then
        pkg update -y && pkg upgrade -y
        log_success "Termux updated"
        
    elif [[ "$PLATFORM" == "linux" ]]; then
        if command -v apt &> /dev/null; then
            sudo apt update && sudo apt upgrade -y
            log_success "APT system updated"
        elif command -v yum &> /dev/null; then
            sudo yum update -y
            log_success "YUM system updated"
        elif command -v pacman &> /dev/null; then
            sudo pacman -Syu --noconfirm
            log_success "Pacman system updated"
        else
            log_warning "Unknown package manager, skipping system update"
        fi
        
    elif [[ "$PLATFORM" == "macos" ]]; then
        brew update && brew upgrade
        log_success "Homebrew updated"
    fi
}

# Install system dependencies
install_system_deps() {
    log_step "Installing system dependencies..."
    
    if [[ "$PLATFORM" == "termux" ]]; then
        log_info "Installing Termux packages..."
        pkg install -y git python python-pip nodejs clang make cmake wget curl
        pkg install -y proot-distro nano vim tree htop neofetch
        pkg install -y rtl-sdr hackrf gr-osmosdr gnuroadio
        pkg install -y tshark wireshark nmap netcat-openbsd
        pkg install -y imagemagick ffmpeg sox
        pkg install -y sqlite mariadb
        pkg install -y binutils radare2 gdb gef
        pkg install -y qemu-system-x86_64
        pkg install -y z3 yices
        pkg install -y verilog gtkwave iverilog
        log_success "Termux packages installed"
        
    elif [[ "$PLATFORM" == "linux" ]]; then
        log_info "Installing Linux packages..."
        
        if command -v apt &> /dev/null; then
            sudo apt install -y \
                git python3 python3-pip python3-venv nodejs npm \
                build-essential clang cmake wget curl \
                rtl-sdr hackrf gr-osmosdr gnuradio \
                tshark wireshark nmap netcat \
                imagemagick ffmpeg sox \
                sqlite3 mysql-client mariadb-client \
                binutils radare2 gdb gef \
                qemu-system-x86 qemu-utils \
                z3 yices \
                iverilog gtkwave verilator \
                libusb-1.0-0 libusb-1.0-0-dev \
                libssl-dev libffi-dev \
                libxml2-dev libxslt1-dev \
                zlib1g-dev libjpeg-dev libpng-dev
            log_success "APT packages installed"
            
        elif command -v yum &> /dev/null; then
            sudo yum install -y \
                git python3 python3-pip nodejs npm \
                gcc gcc-c++ clang cmake wget curl \
                rtl-sdr hackrf gnuradio \
                wireshark nmap netcat \
                ImageMagick ffmpeg sox \
                sqlite mariadb \
                binutils radare2 gdb \
                qemu-system-x86 \
                z3 yices \
                iverilog gtkwave \
                libusb libusb-devel \
                openssl-devel libffi-devel \
                libxml2-devel libxslt-devel \
                zlib-devel libjpeg-turbo-devel libpng-devel
            log_success "YUM packages installed"
        fi
        
    elif [[ "$PLATFORM" == "macos" ]]; then
        log_info "Installing macOS packages..."
        brew install \
            git python node npm \
            cmake wget curl \
            rtl-sdr hackrf gnuradio \
            wireshark nmap netcat \
            imagemagick ffmpeg sox \
            sqlite mysql \
            radare2 gdb \
            qemu \
            z3 yices \
            iverilog gtkwave \
            libusb
        log_success "Homebrew packages installed"
    fi
}

# Setup Python virtual environment
setup_python_env() {
    log_step "Setting up Python environment..."
    
    # Check Python version
    python_version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    if [[ $(echo "$python_version < 3.8" | bc -l) -eq 1 ]]; then
        log_error "Python 3.8+ required. Found: $python_version"
        exit 1
    fi
    log_info "Python version: $python_version"
    
    # Create virtual environment
    if [[ ! -d "venv" ]]; then
        python3 -m venv venv
        log_success "Virtual environment created"
    fi
    
    # Activate virtual environment
    if [[ "$PLATFORM" == "termux" ]]; then
        source venv/bin/activate
    else
        source venv/bin/activate
    fi
    
    # Upgrade pip
    pip install --upgrade pip setuptools wheel
    
    log_success "Python environment ready"
}

# Install Python packages
install_python_packages() {
    log_step "Installing Python packages (this will take a while)..."
    
    # Check if requirements.txt exists
    if [[ ! -f "requirements.txt" ]]; then
        log_error "requirements.txt not found!"
        exit 1
    fi
    
    # Install in batches to handle potential issues
    log_info "Installing core packages..."
    pip install flask flask-socketio flask-limiter flask-cors
    
    log_info "Installing cryptography packages..."
    pip install cryptography pycryptodome ecdsa rsa argon2-cffi
    
    log_info "Installing quantum packages..."
    pip install qiskit qiskit-aer qiskit-ibm-runtime
    
    log_info "Installing analysis packages..."
    pip install numpy pandas scipy matplotlib sympy
    
    log_info "Installing remaining packages..."
    # Try to install everything, but continue on error
    pip install -r requirements.txt || {
        log_warning "Some packages failed to install. Continuing..."
    }
    
    log_success "Python packages installed"
}

# Install Node.js packages
install_node_packages() {
    log_step "Installing Node.js packages..."
    
    if command -v npm &> /dev/null; then
        npm install -g express socket.io threejs webpack
        npm install -g minecraft-server
        log_success "Node.js packages installed"
    else
        log_warning "npm not found, skipping Node.js packages"
    fi
}

# Setup database
setup_database() {
    log_step "Setting up databases..."
    
    # SQLite setup
    if [[ ! -f "chimera.db" ]]; then
        sqlite3 chimera.db ".databases"
        log_success "SQLite database created"
    fi
    
    # MySQL/MariaDB setup (if available)
    if command -v mysql &> /dev/null; then
        log_info "Setting up MySQL user..."
        # This is just for development
        # In production, you'd have proper setup
        echo "MySQL setup would go here"
    fi
    
    log_success "Databases ready"
}

# Setup directories
setup_directories() {
    log_step "Setting up directories..."
    
    # Create necessary directories
    mkdir -p logs
    mkdir -p data
    mkdir -p temp
    mkdir -p keys
    mkdir -p uploads
    mkdir -p downloads
    mkdir -p backups
    
    # Set permissions
    chmod 755 logs data temp uploads downloads backups
    chmod 700 keys
    
    log_success "Directories created"
}

# Generate encryption keys
generate_keys() {
    log_step "Generating encryption keys..."
    
    if [[ ! -d "keys" ]]; then
        mkdir -p keys
    fi
    
    # Generate server key
    if [[ ! -f "keys/server.key" ]]; then
        openssl genrsa -out keys/server.key 4096 2>/dev/null || \
        python3 -c "
import secrets
key = secrets.token_hex(256)
with open('keys/server.key', 'w') as f:
    f.write(key)
        "
        chmod 600 keys/server.key
        log_success "Server key generated"
    fi
    
    # Generate session secret
    if [[ ! -f "keys/session.secret" ]]; then
        python3 -c "
import secrets
import base64
secret = base64.b64encode(secrets.token_bytes(32)).decode()
with open('keys/session.secret', 'w') as f:
    f.write(secret)
        "
        chmod 600 keys/session.secret
        log_success "Session secret generated"
    fi
    
    log_success "Encryption keys ready"
}

# Hardware detection
detect_hardware() {
    log_step "Detecting hardware capabilities..."
    
    # Create hardware profile
    HARDWARE_FILE="data/hardware_profile.json"
    
    python3 -c "
import platform
import psutil
import json
import os

def get_hardware_info():
    info = {
        'platform': {
            'system': platform.system(),
            'release': platform.release(),
            'version': platform.version(),
            'machine': platform.machine(),
            'processor': platform.processor()
        },
        'cpu': {
            'cores': psutil.cpu_count(logical=False),
            'threads': psutil.cpu_count(logical=True),
            'frequency': psutil.cpu_freq().current if psutil.cpu_freq() else None
        },
        'memory': {
            'total': psutil.virtual_memory().total,
            'available': psutil.virtual_memory().available
        },
        'disk': {
            'total': psutil.disk_usage('/').total,
            'free': psutil.disk_usage('/').free
        },
        'python': {
            'version': platform.python_version(),
            'implementation': platform.python_implementation()
        }
    }
    return info

info = get_hardware_info()
with open('$HARDWARE_FILE', 'w') as f:
    json.dump(info, f, indent=2)
print(f'Hardware profile saved to $HARDWARE_FILE')
    "
    
    log_success "Hardware profile created"
}

# Setup monitoring
setup_monitoring() {
    log_step "Setting up monitoring..."
    
    # Create monitoring script
    cat > monitoring.sh << 'EOF'
#!/bin/bash
# Chimera-VX Monitoring Script

while true; do
    TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
    LOG_FILE="logs/monitor_${TIMESTAMP}.log"
    
    echo "=== CHIMERA-VX MONITORING REPORT ===" > $LOG_FILE
    echo "Timestamp: $TIMESTAMP" >> $LOG_FILE
    echo "" >> $LOG_FILE
    
    # CPU usage
    echo "CPU Usage:" >> $LOG_FILE
    top -bn1 | grep "Cpu(s)" | sed "s/.*, *\([0-9.]*\)%* id.*/\1/" | awk '{print "  Idle: " 100 - $1 "%"}' >> $LOG_FILE
    echo "" >> $LOG_FILE
    
    # Memory usage
    echo "Memory Usage:" >> $LOG_FILE
    free -h | awk 'NR==2{print "  Used: " $3 "/" $2}' >> $LOG_FILE
    echo "" >> $LOG_FILE
    
    # Disk usage
    echo "Disk Usage:" >> $LOG_FILE
    df -h / | awk 'NR==2{print "  Used: " $3 "/" $2 " (" $5 ")"}' >> $LOG_FILE
    echo "" >> $LOG_FILE
    
    # Process list
    echo "Top Processes:" >> $LOG_FILE
    ps aux --sort=-%cpu | head -6 >> $LOG_FILE
    echo "" >> $LOG_FILE
    
    sleep 300  # Run every 5 minutes
done
EOF
    
    chmod +x monitoring.sh
    
    log_success "Monitoring setup complete"
}

# Final setup
final_setup() {
    log_step "Finalizing setup..."
    
    # Create first-run flag
    echo "CHIMERA-VX INSTALLED $(date)" > .chimera_installed
    
    # Set permissions
    chmod +x server/*.py client/*.py
    
    log_success "Setup complete!"
}

# Print completion message
print_completion() {
    echo -e "${GREEN}"
    echo "=================================================================================="
    echo "                            SETUP COMPLETE!"
    echo "=================================================================================="
    echo -e "${NC}"
    
    log_chimera "Chimera-VX has been installed successfully!"
    echo ""
    log_info "Next steps:"
    echo "  1. Review the configuration in server/config.py"
    echo "  2. Start the server: python3 server/main_server.py"
    echo "  3. Run the client: python3 client/player_client.py"
    echo ""
    log_warning "IMPORTANT: Keep your keys/ directory secure!"
    echo ""
    log_info "Hardware profile saved to: data/hardware_profile.json"
    log_info "Logs will be stored in: logs/"
    log_info "Temporary files in: temp/"
    echo ""
    
    # Print hardware summary
    if [[ -f "data/hardware_profile.json" ]]; then
        echo -e "${CYAN}HARDWARE SUMMARY:${NC}"
        python3 -c "
import json
with open('data/hardware_profile.json') as f:
    data = json.load(f)
print(f'  CPU: {data[\"cpu\"][\"cores\"]} cores, {data[\"cpu\"][\"threads\"]} threads')
print(f'  RAM: {data[\"memory\"][\"total\"] // (1024**3)} GB total')
print(f'  Disk: {data[\"disk\"][\"total\"] // (1024**3)} GB total')
        "
    fi
    
    echo ""
    echo -e "${RED}=================================================================================="
    echo "                          WARNING: THE CHALLENGE BEGINS"
    echo "=================================================================================="
    echo -e "${NC}"
}

# Main execution
main() {
    print_banner
    check_root
    detect_platform
    
    log_chimera "Starting Chimera-VX installation..."
    echo ""
    
    # Run setup steps
    update_system
    install_system_deps
    setup_python_env
    install_python_packages
    install_node_packages
    setup_database
    setup_directories
    generate_keys
    detect_hardware
    setup_monitoring
    final_setup
    
    print_completion
}

# Run main function
main "$@"