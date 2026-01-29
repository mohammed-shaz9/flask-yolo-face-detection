import os
import sys
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure 'src' is in path so imports work when running from root
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, send_from_directory, jsonify

app = Flask(__name__, static_folder=os.path.join(os.path.dirname(__file__), 'static'))
app.config['SECRET_KEY'] = 'asdf#FGSgvasgf$5$WGT'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

# Register blueprint with error handling
try:
    from routes.face_detection import face_detection_bp
    app.register_blueprint(face_detection_bp, url_prefix='/api')
    logger.info("Face detection blueprint registered successfully")
except Exception as e:
    logger.error(f"Failed to register face detection blueprint: {e}")
    import traceback
    logger.error(traceback.format_exc())


@app.route('/health')
def health():
    """Root health check"""
    return jsonify({
        'status': 'ok',
        'message': 'Flask app is running'
    })


@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    static_folder_path = app.static_folder
    if static_folder_path is None:
        return "Static folder not configured", 404

    if path != "" and os.path.exists(os.path.join(static_folder_path, path)):
        return send_from_directory(static_folder_path, path)
    else:
        index_path = os.path.join(static_folder_path, 'index.html')
        if os.path.exists(index_path):
            return send_from_directory(static_folder_path, 'index.html')
        else:
            return "index.html not found", 404


@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {error}")
    return jsonify({'success': False, 'error': 'Internal server error'}), 500


@app.errorhandler(404)
def not_found(error):
    return jsonify({'success': False, 'error': 'Not found'}), 404


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 7860))
    logger.info(f"Starting Flask app on port {port}")
    app.run(host='0.0.0.0', port=port, debug=True)
