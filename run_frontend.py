import http.server
import socketserver
import os
import webbrowser
from pathlib import Path

# Change to frontend directory
frontend_dir = Path("frontend")
if frontend_dir.exists():
    os.chdir(frontend_dir)
    
    PORT = 3000
    Handler = http.server.SimpleHTTPRequestHandler
    
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        print(f"🌐 Frontend server running at: http://localhost:{PORT}")
        print(f"🔗 Backend API running at: http://127.0.0.1:8001")
        print("🚀 Opening browser...")
        
        # Open browser automatically
        webbrowser.open(f"http://localhost:{PORT}")
        
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n👋 Frontend server stopped")
else:
    print("❌ Frontend directory not found")
