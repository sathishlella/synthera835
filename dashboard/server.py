"""
SynthERA-835 Dashboard API Server

Flask server that serves the dashboard and provides API endpoints
to run the Python generator directly from the browser.
"""

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import subprocess
import threading
import os
import sys
from pathlib import Path

# Get paths correctly
# server.py is in synthera835/dashboard/
# So parent is synthera835/, parent.parent is Velden_Research/
DASHBOARD_DIR = Path(__file__).parent  # synthera835/dashboard
SYNTHERA_DIR = DASHBOARD_DIR.parent     # synthera835
PROJECT_ROOT = SYNTHERA_DIR.parent      # Velden_Research
OUTPUT_DIR = PROJECT_ROOT / 'synthera835_output'

app = Flask(__name__, static_folder=str(DASHBOARD_DIR))
CORS(app)  # Enable CORS for development

# Track generation status
generation_status = {
    'running': False,
    'progress': 0,
    'message': '',
    'error': None,
    'result': None
}


@app.route('/')
def index():
    """Serve the dashboard."""
    return send_from_directory(app.static_folder, 'index.html')


@app.route('/<path:path>')
def static_files(path):
    """Serve static files."""
    # First check if file exists in static folder
    static_path = Path(app.static_folder) / path
    if static_path.exists():
        return send_from_directory(app.static_folder, path)
    
    # API routes are handled by specific decorators, so if we get here for non-static
    # return 404
    return "Not found", 404


@app.route('/api/generate', methods=['POST'])
def generate_claims():
    """
    Generate claims using the Python generator.
    
    Request body:
    {
        "num_claims": 100,
        "denial_rate": 0.11,
        "seed": null
    }
    """
    global generation_status
    
    if generation_status['running']:
        return jsonify({
            'success': False,
            'error': 'Generation already in progress'
        }), 409
    
    data = request.get_json() or {}
    num_claims = data.get('num_claims', 100)
    denial_rate = data.get('denial_rate', 0.11)
    seed = data.get('seed', None)
    
    # Start generation in background thread
    thread = threading.Thread(
        target=run_generator,
        args=(num_claims, denial_rate, seed)
    )
    thread.start()
    
    return jsonify({
        'success': True,
        'message': f'Started generating {num_claims} claims'
    })


def run_generator(num_claims, denial_rate, seed):
    """Run the generator in a background thread."""
    global generation_status
    
    generation_status = {
        'running': True,
        'progress': 0,
        'message': f'Generating {num_claims} claims...',
        'error': None,
        'result': None
    }
    
    try:
        # Build the command to run the generator
        seed_str = str(seed) if seed else 'None'
        cmd = [
            sys.executable, '-c', f'''
import sys
# Add the directory containing generator.py to python path
sys.path.insert(0, r"{SYNTHERA_DIR}")

from generator import ERA835Generator

generator = ERA835Generator(
    carc_csv_path=None,
    rarc_csv_path=None,
    denial_rate={denial_rate},
    seed={seed_str}
)

# Use raw string for path to avoid escape issues on Windows
# Only pass output_dir if needed
claims = generator.generate_dataset(
    num_claims={num_claims},
    output_dir=r"{OUTPUT_DIR}",
    include_edi=False
)

stats = generator.get_statistics()
print(f"Generated {{stats['claims_generated']}} claims")
print(f"Denied: {{stats['claims_denied']}}")
print(f"Paid: {{stats['claims_paid']}}")
'''
        ]
        
        # Run the generator
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
            timeout=300  # 5 minute timeout
        )
        
        if result.returncode != 0:
            generation_status['error'] = result.stderr
            generation_status['message'] = 'Generation failed'
            print(f"Generator error: {result.stderr}")
        else:
            generation_status['result'] = result.stdout
            generation_status['message'] = 'Generation complete'
            generation_status['progress'] = 100
            print(f"Generator output: {result.stdout}")
        
    except subprocess.TimeoutExpired:
        generation_status['error'] = 'Generation timed out'
        generation_status['message'] = 'Generation timed out'
    except Exception as e:
        generation_status['error'] = str(e)
        generation_status['message'] = f'Error: {str(e)}'
        print(f"Exception: {e}")
    finally:
        generation_status['running'] = False


@app.route('/api/status', methods=['GET'])
def get_status():
    """Get the current generation status."""
    return jsonify(generation_status)


@app.route('/api/claims', methods=['GET'])
def get_claims():
    """Get the generated claims data."""
    claims_file = OUTPUT_DIR / 'claims.csv'
    
    if not claims_file.exists():
        return jsonify({
            'success': False,
            'error': 'No claims data found'
        }), 404
    
    # Read and return the CSV content
    content = claims_file.read_text()
    return content, 200, {'Content-Type': 'text/csv'}


@app.route('/synthera835_output/<path:path>')
def serve_output(path):
    """Serve files from the output directory."""
    return send_from_directory(str(OUTPUT_DIR), path)


if __name__ == '__main__':
    print(f"=" * 50)
    print(f"SynthERA-835 Dashboard Server")
    print(f"=" * 50)
    print(f"Dashboard dir: {DASHBOARD_DIR}")
    print(f"SynthERA dir:  {SYNTHERA_DIR}")
    print(f"Project root:  {PROJECT_ROOT}")
    print(f"Output dir:    {OUTPUT_DIR}")
    print(f"Static folder: {app.static_folder}")
    print(f"\n✓ Open http://localhost:5000 in your browser")
    print(f"=" * 50)
    
    # Ensure output directory exists
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    app.run(host='0.0.0.0', port=5000, debug=True)
