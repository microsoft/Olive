# Olive UI

A web interface for Microsoft Olive - AI Model Optimization Toolkit.

## 📋 Prerequisites

- Python 3.10+ with Olive installed
- Node.js 14+ and npm
- Windows (for .bat scripts)

## 🚀 Setup Instructions

### Step 1: Install Backend Dependencies

```bash
cd server
pip install -r requirements.txt
cd ..
```

### Step 2: Install Frontend Dependencies

```bash
cd client
npm install
cd ..
```

### Step 3: Run the Application

```bash
# Just double-click or run:
start.bat
```

That's it! The app will:
- Clean up any existing processes on ports 3000/8000  
- Start the backend server
- Start the frontend
- Open in your browser at http://localhost:3000

## 🔧 Custom Ports

```bash
# Use custom ports (frontend_port backend_port)
start.bat 3001 8001
```

## ✨ Features

- **Pass Configuration**: Configure and run Olive optimization passes
- **Job Monitor**: Monitor workflow progress with collapsible sections  
- **CLI Interface**: Run Olive CLI commands through the web interface
- **Single Window**: Everything runs in one terminal (no more popup windows!)

## 🔧 Manual Start

If you prefer to start the servers manually:

#### Backend Server
```bash
cd server
python app.py --port 8000
```

#### Frontend
```bash
cd client
npm start
```

The UI will open in your browser at http://localhost:3000

## 🚀 Usage

### Pass Configuration
Configure and run Olive optimization workflows with visual interface.

### Job Monitor  
Monitor workflow progress with collapsible sections for easy navigation.

### CLI Interface
Run Olive CLI commands through the web interface.

## 🛠️ Troubleshooting

If you get port conflicts, the startup script will automatically kill existing processes on the required ports.

For manual cleanup:
```bash
kill-ports.bat
```

## 🏗️ Architecture

- **Backend**: FastAPI with async support
- **Frontend**: React TypeScript application
- **State Management**: React hooks with sessionStorage persistence  
- **API Communication**: Axios for REST, WebSocket for real-time updates

## API Endpoints

- `GET /api/passes` - Get all available passes categorized
- `GET /api/pass/{pass_name}/schema` - Get configuration schema for a pass
- `GET /api/cli-commands` - Get all CLI commands and their arguments
- `POST /api/workflow/validate` - Validate a workflow configuration
- `POST /api/workflow/run` - Run a workflow
- `POST /api/cli/run` - Run a CLI command
- `GET /api/job/{job_id}` - Get job status
- `WS /ws/{job_id}` - WebSocket for real-time job updates