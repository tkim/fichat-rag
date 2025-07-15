# fiChat-RAG Local Setup Guide

This guide provides step-by-step instructions for running fiChat-RAG locally on different platforms: Windows (native), Windows Subsystem for Linux (WSL), and Google Colab.

## Table of Contents

1. [Windows Native Setup](#windows-native-setup)
2. [Windows Subsystem for Linux (WSL)](#windows-subsystem-for-linux-wsl)
3. [Google Colab Setup](#google-colab-setup)
4. [Troubleshooting](#troubleshooting)

---

## Windows Native Setup

### Prerequisites

1. **Python 3.8+**
   - Download from [python.org](https://www.python.org/downloads/)
   - During installation, check "Add Python to PATH"

2. **Git**
   - Download from [git-scm.com](https://git-scm.com/download/win)

3. **Ollama**
   - Download from [ollama.ai](https://ollama.ai/download/windows)
   - Run the installer and follow the prompts

4. **Visual Studio Build Tools** (for some Python packages)
   - Download from [Microsoft](https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022)
   - Install "Desktop development with C++"

### Step-by-Step Installation

1. **Clone the Repository**
   ```powershell
   # Open PowerShell or Command Prompt
   git clone https://github.com/tkim/fichat-rag.git
   cd fichat-rag
   ```

2. **Create Virtual Environment**
   ```powershell
   # Create virtual environment
   python -m venv venv
   
   # Activate virtual environment
   # For PowerShell:
   .\venv\Scripts\Activate.ps1
   
   # For Command Prompt:
   venv\Scripts\activate.bat
   ```

3. **Install Dependencies**
   ```powershell
   # Upgrade pip
   python -m pip install --upgrade pip
   
   # Install requirements
   pip install -r requirements.txt
   
   # For SQLite vector support (optional)
   pip install sqlite-vss
   ```

4. **Start Ollama Service**
   ```powershell
   # In a new terminal window
   ollama serve
   
   # Pull a model (in another terminal)
   ollama pull llama2
   # Or for a smaller model:
   ollama pull phi
   ```

5. **Configure Environment**
   ```powershell
   # Copy example environment file
   copy .env.example .env
   
   # Edit .env file (use notepad or your preferred editor)
   notepad .env
   ```

   Update the `.env` file:
   ```env
   # LLM Configuration
   LLM_PROVIDER=ollama
   LLM_MODEL=llama2
   LLM_BASE_URL=http://localhost:11434
   
   # Vector Store Configuration
   VECTOR_STORE_TYPE=sqlite
   SQLITE_DB_PATH=./data/fichat.db
   
   # Embeddings Configuration
   EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
   EMBEDDING_DEVICE=cpu
   ```

6. **Create Required Directories**
   ```powershell
   # Create data and documents directories
   mkdir data
   mkdir documents
   ```

7. **Run the Application**
   ```powershell
   # Run the example script
   python examples/ollama_local.py
   
   # Or start the API server
   python -m uvicorn src.api:app --reload --host 0.0.0.0 --port 8000
   ```

### Windows-Specific Considerations

1. **Firewall Settings**
   - Windows Firewall may prompt for access when running Ollama
   - Allow access for both Private and Public networks

2. **Long Path Support**
   - Enable long path support if you encounter path length issues:
   ```powershell
   # Run as Administrator
   New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
   ```

3. **GPU Support (NVIDIA)**
   - Install CUDA Toolkit from [NVIDIA](https://developer.nvidia.com/cuda-downloads)
   - Install cuDNN
   - Update `.env` to use GPU:
   ```env
   EMBEDDING_DEVICE=cuda
   ```

---

## Windows Subsystem for Linux (WSL)

### Prerequisites

1. **Install WSL2**
   ```powershell
   # Run as Administrator in PowerShell
   wsl --install
   
   # Restart your computer
   
   # Set WSL2 as default
   wsl --set-default-version 2
   
   # Install Ubuntu (or your preferred distro)
   wsl --install -d Ubuntu-22.04
   ```

2. **Update WSL Ubuntu**
   ```bash
   # Inside WSL terminal
   sudo apt update && sudo apt upgrade -y
   ```

### Step-by-Step Installation

1. **Install System Dependencies**
   ```bash
   # Install Python and development tools
   sudo apt install -y python3-pip python3-venv python3-dev
   sudo apt install -y build-essential git curl wget
   
   # Install SQLite with extensions support
   sudo apt install -y sqlite3 libsqlite3-dev
   ```

2. **Install Ollama in WSL**
   ```bash
   # Download and install Ollama
   curl -fsSL https://ollama.ai/install.sh | sh
   
   # Start Ollama service
   ollama serve &
   
   # Pull a model
   ollama pull llama2
   ```

3. **Clone and Setup fiChat-RAG**
   ```bash
   # Clone repository
   git clone https://github.com/tkim/fichat-rag.git
   cd fichat-rag
   
   # Create virtual environment
   python3 -m venv venv
   source venv/bin/activate
   
   # Install dependencies
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Configure for WSL**
   ```bash
   # Copy and edit environment file
   cp .env.example .env
   nano .env
   ```

   WSL-specific configuration:
   ```env
   # LLM Configuration
   LLM_PROVIDER=ollama
   LLM_MODEL=llama2
   LLM_BASE_URL=http://localhost:11434
   
   # Vector Store Configuration
   VECTOR_STORE_TYPE=sqlite
   SQLITE_DB_PATH=/home/$USER/fichat-rag/data/fichat.db
   
   # Embeddings Configuration
   EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
   EMBEDDING_DEVICE=cpu
   ```

5. **Create Systemd Service (Optional)**
   ```bash
   # Create service file
   sudo nano /etc/systemd/system/ollama.service
   ```

   Add:
   ```ini
   [Unit]
   Description=Ollama Service
   After=network.target
   
   [Service]
   Type=simple
   User=$USER
   ExecStart=/usr/local/bin/ollama serve
   Restart=always
   
   [Install]
   WantedBy=multi-user.target
   ```

   Enable service:
   ```bash
   sudo systemctl enable ollama
   sudo systemctl start ollama
   ```

6. **Run the Application**
   ```bash
   # Activate virtual environment
   source venv/bin/activate
   
   # Run example
   python examples/ollama_local.py
   
   # Or start API server
   python -m uvicorn src.api:app --reload --host 0.0.0.0 --port 8000
   ```

### WSL-Specific Features

1. **GPU Passthrough (NVIDIA)**
   ```bash
   # Check if GPU is available in WSL
   nvidia-smi
   
   # If available, update .env
   EMBEDDING_DEVICE=cuda
   ```

2. **Access from Windows**
   - API will be accessible at `http://localhost:8000` from Windows
   - Files can be accessed at `\\wsl$\Ubuntu-22.04\home\<username>\fichat-rag`

3. **Memory Configuration**
   Create `.wslconfig` in Windows user directory:
   ```ini
   [wsl2]
   memory=8GB
   processors=4
   swap=2GB
   ```

---

## Google Colab Setup

### Quick Start Notebook

1. **Create New Colab Notebook**
   - Go to [Google Colab](https://colab.research.google.com/)
   - Create a new notebook

2. **Complete Setup Script**
   ```python
   # Cell 1: Install Ollama and dependencies
   !curl -fsSL https://ollama.ai/install.sh | sh
   !pip install -q requests aiohttp numpy sentence-transformers
   !pip install -q psycopg2-binary sqlalchemy chromadb
   
   # Cell 2: Clone fiChat-RAG
   !git clone https://github.com/tkim/fichat-rag.git
   %cd fichat-rag
   !pip install -q -r requirements.txt
   
   # Cell 3: Start Ollama in background
   import subprocess
   import time
   
   # Start Ollama server
   ollama_process = subprocess.Popen(['ollama', 'serve'])
   time.sleep(5)  # Wait for server to start
   
   # Pull a model
   !ollama pull phi  # Using phi for faster download in Colab
   
   # Cell 4: Configure environment
   import os
   
   os.environ['LLM_PROVIDER'] = 'ollama'
   os.environ['LLM_MODEL'] = 'phi'
   os.environ['LLM_BASE_URL'] = 'http://localhost:11434'
   os.environ['VECTOR_STORE_TYPE'] = 'memory'  # Using memory for Colab
   os.environ['EMBEDDING_MODEL'] = 'sentence-transformers/all-MiniLM-L6-v2'
   os.environ['EMBEDDING_DEVICE'] = 'cuda' if torch.cuda.is_available() else 'cpu'
   
   # Cell 5: Import and initialize fiChat-RAG
   import sys
   sys.path.append('/content/fichat-rag')
   
   from src import RAG, Config
   from src.storage.base import Document
   
   # Initialize configuration
   config = Config()
   rag = RAG(config=config)
   
   # Cell 6: Add sample documents
   sample_docs = [
       Document(
           content="Google Colab provides free GPU resources for machine learning.",
           metadata={"source": "colab_info"}
       ),
       Document(
           content="Ollama enables running LLMs locally with simple commands.",
           metadata={"source": "ollama_info"}
       )
   ]
   
   rag.add_documents(sample_docs)
   print("Documents added successfully!")
   
   # Cell 7: Query the system
   question = "What is Google Colab?"
   response = rag.query(question)
   print(f"Question: {question}")
   print(f"Answer: {response}")
   ```

### Advanced Colab Setup with Persistent Storage

```python
# Cell 1: Mount Google Drive for persistence
from google.colab import drive
drive.mount('/content/drive')

# Create directory for fiChat-RAG data
!mkdir -p /content/drive/MyDrive/fichat-rag-data

# Cell 2: Install with persistent storage
!cd /content/drive/MyDrive && git clone https://github.com/tkim/fichat-rag.git
%cd /content/drive/MyDrive/fichat-rag
!pip install -q -r requirements.txt

# Cell 3: Setup SQLite with persistent storage
import os
os.environ['SQLITE_DB_PATH'] = '/content/drive/MyDrive/fichat-rag-data/fichat.db'
os.environ['VECTOR_STORE_TYPE'] = 'sqlite'

# Cell 4: Create Colab-specific utilities
%%writefile colab_utils.py
import subprocess
import time
import requests
import os

class ColabOllamaManager:
    def __init__(self):
        self.process = None
        
    def start(self):
        """Start Ollama server"""
        self.process = subprocess.Popen(['ollama', 'serve'])
        time.sleep(5)
        
    def stop(self):
        """Stop Ollama server"""
        if self.process:
            self.process.terminate()
            
    def is_running(self):
        """Check if Ollama is running"""
        try:
            response = requests.get('http://localhost:11434/api/tags')
            return response.status_code == 200
        except:
            return False
            
    def pull_model(self, model_name='phi'):
        """Pull a model"""
        os.system(f'ollama pull {model_name}')

# Cell 5: Create interactive interface
import ipywidgets as widgets
from IPython.display import display, HTML

class ChatInterface:
    def __init__(self, rag):
        self.rag = rag
        self.setup_ui()
        
    def setup_ui(self):
        self.query_input = widgets.Text(
            placeholder='Ask a question...',
            layout=widgets.Layout(width='70%')
        )
        self.submit_button = widgets.Button(
            description='Ask',
            button_style='primary'
        )
        self.output = widgets.Output()
        
        self.submit_button.on_click(self.on_submit)
        
        display(widgets.HBox([self.query_input, self.submit_button]))
        display(self.output)
        
    def on_submit(self, _):
        with self.output:
            question = self.query_input.value
            if question:
                print(f"Q: {question}")
                response = self.rag.query(question)
                print(f"A: {response}\n")
                self.query_input.value = ""

# Usage
chat = ChatInterface(rag)
```

### Colab-Specific Optimizations

1. **GPU Acceleration**
   ```python
   # Check GPU availability
   import torch
   print(f"GPU Available: {torch.cuda.is_available()}")
   if torch.cuda.is_available():
       print(f"GPU Name: {torch.cuda.get_device_name(0)}")
   ```

2. **Memory Management**
   ```python
   # Clear GPU memory when needed
   import gc
   import torch
   
   def clear_gpu_memory():
       gc.collect()
       if torch.cuda.is_available():
           torch.cuda.empty_cache()
   ```

3. **Download Models to Drive**
   ```python
   # Save Ollama models to Google Drive
   !mkdir -p /content/drive/MyDrive/ollama-models
   !ln -s /content/drive/MyDrive/ollama-models ~/.ollama
   ```

### Colab Notebook Template

Create a ready-to-use notebook:

```python
# Cell 1: Complete Setup
!curl -fsSL https://ollama.ai/install.sh | sh > /dev/null 2>&1
!git clone https://github.com/tkim/fichat-rag.git > /dev/null 2>&1
%cd fichat-rag
!pip install -q -r requirements.txt

import subprocess
import time
import os
import sys

# Start Ollama
ollama_proc = subprocess.Popen(['ollama', 'serve'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
time.sleep(5)

# Pull model
!ollama pull phi > /dev/null 2>&1

# Configure
os.environ.update({
    'LLM_PROVIDER': 'ollama',
    'LLM_MODEL': 'phi',
    'VECTOR_STORE_TYPE': 'memory',
    'EMBEDDING_MODEL': 'sentence-transformers/all-MiniLM-L6-v2'
})

sys.path.append('/content/fichat-rag')
from src import RAG, Config

print("✅ Setup complete! fiChat-RAG is ready to use.")

# Cell 2: Initialize and Test
config = Config()
rag = RAG(config=config)

# Add a test document
from src.storage.base import Document
rag.add_documents([
    Document(
        content="This is a test document in Google Colab.",
        metadata={"source": "test"}
    )
])

# Test query
response = rag.query("What is this document about?")
print(f"Test Response: {response}")

# Cell 3: Interactive Usage
def ask(question):
    """Simple function to query the RAG system"""
    return rag.query(question)

# Now you can use: ask("Your question here")
```

---

## Troubleshooting

### Common Issues and Solutions

#### Windows Issues

1. **PowerShell Execution Policy**
   ```powershell
   # If you can't run scripts
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
   ```

2. **Python Package Installation Errors**
   - Install Visual Studio Build Tools
   - Use pre-compiled wheels: `pip install --only-binary :all: package_name`

3. **Ollama Connection Refused**
   - Check Windows Defender Firewall
   - Ensure Ollama service is running: `ollama serve`

#### WSL Issues

1. **WSL2 Memory Limits**
   - Create `.wslconfig` in Windows user directory
   - Allocate more memory to WSL2

2. **Network Issues**
   ```bash
   # If localhost doesn't work, use WSL IP
   ip addr show eth0 | grep -oP '(?<=inet\s)\d+(\.\d+){3}'
   ```

3. **Permission Errors**
   ```bash
   # Fix permissions
   sudo chown -R $USER:$USER ~/fichat-rag
   ```

#### Google Colab Issues

1. **Runtime Disconnections**
   - Use Google Drive for persistent storage
   - Save checkpoints regularly

2. **Memory Errors**
   ```python
   # Clear memory
   import gc
   gc.collect()
   
   # For GPU
   torch.cuda.empty_cache()
   ```

3. **Model Download Timeouts**
   - Use smaller models (phi, tinyllama)
   - Pre-download to Google Drive

### Performance Tips

1. **Windows Native**
   - Use conda instead of pip for better package management
   - Enable GPU acceleration if available

2. **WSL**
   - Increase WSL2 memory allocation
   - Use WSL2 GPU passthrough for CUDA

3. **Google Colab**
   - Use GPU runtime when available
   - Cache models in Google Drive
   - Use batch processing for large datasets

### Getting Help

1. Check the [GitHub Issues](https://github.com/tkim/fichat-rag/issues)
2. Review [Ollama Documentation](https://github.com/jmorganca/ollama)
3. Join our [Discord Community](#) (if available)

---

## Quick Reference Commands

### Windows PowerShell
```powershell
# Start Ollama
ollama serve

# Pull model
ollama pull llama2

# Run fiChat-RAG
python examples/ollama_local.py
```

### WSL/Linux
```bash
# Start Ollama
ollama serve &

# Pull model
ollama pull llama2

# Run fiChat-RAG
source venv/bin/activate
python examples/ollama_local.py
```

### Google Colab
```python
# One-line setup
!curl -fsSL https://ollama.ai/install.sh | sh && ollama serve & && ollama pull phi

# Import and use
from src import RAG, Config
rag = RAG(Config())
```